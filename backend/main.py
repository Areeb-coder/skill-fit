"""
AI SkillFit - Complete FastAPI Backend
All endpoints for candidate flow + admin dashboard
"""
import os
import json
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn

#from config import settings
#import database as db
#from core.language_manager import get_sms_template, validate_language
#from core.fraud_engine import get_fraud_engine
#from core.assessment_engine import get_assessment_engine

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="AI SkillFit API", version="2.0.0")

# CORS - Allow frontend origin
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded files
os.makedirs("./uploads", exist_ok=True)
os.makedirs("./uploads/photos", exist_ok=True)
os.makedirs("./uploads/videos", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="./uploads"), name="uploads")

security = HTTPBearer(auto_error=False)

# ─── Auth ─────────────────────────────────────────────────────────────────────

def create_token(email: str) -> str:
    import jwt
    payload = {"sub": email, "exp": datetime.utcnow() + timedelta(days=7)}
    return jwt.encode(payload, settings.secret_key, algorithm="HS256")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        import jwt
        payload = jwt.decode(credentials.credentials, settings.secret_key, algorithms=["HS256"])
        return payload["sub"]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"service": "AI SkillFit API", "version": "2.0", "status": "running"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "database": "supabase" if db.get_supabase() else "in-memory",
        "models": ["Whisper", "Sentence Transformers", "MediaPipe", "DeepFace", "librosa"]
    }

# ─── Candidate Registration ───────────────────────────────────────────────────

@app.post("/api/candidate/register")
async def register_candidate(
    name: str = Form(...),
    phone: str = Form(...),
    district: str = Form(...),
    role: str = Form(...),
    language: str = Form(...),
    education: str = Form(""),
    experience_years: int = Form(0),
    photo: UploadFile = File(...)
):
    """Register candidate with face verification"""

    # Validate language
    if not validate_language(language):
        raise HTTPException(status_code=400, detail="Invalid language")

    # Check duplicate phone
    existing = db.get_candidate_by_phone(phone)
    if existing:
        raise HTTPException(status_code=409, detail="phone_exists")

    # Save photo
    photo_filename = f"{uuid.uuid4()}_{photo.filename}"
    photo_path = f"./uploads/photos/{photo_filename}"
    with open(photo_path, "wb") as f:
        f.write(await photo.read())

    # Fraud check on photo
    fraud_engine = get_fraud_engine()
    existing_encodings = db.get_all_face_encodings()
    fraud_result = fraud_engine.check_registration_photo(photo_path, "", existing_encodings)

    if fraud_result.action == "BLOCK":
        os.remove(photo_path)
        if "duplicate" in " ".join(fraud_result.fraud_flags).lower():
            raise HTTPException(status_code=409, detail="face_exists")
        raise HTTPException(status_code=400, detail="no_face_detected")

    # Create candidate
    candidate = db.create_candidate({
        "name": name,
        "phone": phone,
        "district": district,
        "role": role,
        "language": language,
        "education": education,
        "experience_years": experience_years,
        "photo_url": f"/uploads/photos/{photo_filename}"
    })

    # Save face encoding
    face_encoding = fraud_engine.get_face_encoding(photo_path)
    db.save_face_encoding(candidate["id"], photo_path, face_encoding)

    # Create interview session
    session = db.create_session(candidate["id"], language)

    return {
        "success": True,
        "candidate_id": candidate["id"],
        "session_id": session["id"],
        "candidate_name": name
    }

# ─── Interview Questions ──────────────────────────────────────────────────────

@app.get("/api/interview/{session_id}/questions")
async def get_questions(session_id: str):
    """Get all 7 questions for the session in the correct language"""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    language = session["language"]
    candidate = db.get_candidate_by_id(session["candidate_id"])
    role = candidate["applied_role"] if candidate else "electrician"

    questions = _load_questions(role, language)
    return {"questions": questions, "language": language, "total": len(questions)}

def _load_questions(role: str, language: str) -> List[Dict]:
    """Load 7 questions for role and language"""
    # Map language to key
    lang_key = {"kannada": "kannada", "hindi": "hindi", "english": "english"}.get(language, "kannada")

    # Try role-specific file first
    question_file = f"../public/data/questions-{role}.json"
    if not os.path.exists(question_file):
        question_file = f"../public/data/questions-electrician.json"

    try:
        with open(question_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        lang_data = data.get("languages", {}).get(lang_key, {})

        warmup = lang_data.get("warmup", [])[:2]
        easy = lang_data.get("technical_easy", [])[:2]
        medium = lang_data.get("technical_medium", [])[:2]
        situational = lang_data.get("situational", [])[:1]

        questions = warmup + easy + medium + situational

        # Pad to 7 if needed
        while len(questions) < 7 and warmup:
            questions.append(warmup[0])

        return questions[:7]
    except Exception as e:
        print(f"Error loading questions: {e}")
        return _get_fallback_questions(language)

def _get_fallback_questions(language: str) -> List[Dict]:
    """Fallback questions if file not found"""
    questions = {
        "kannada": [
            {"id": "f1", "text": "ನಮಸ್ಕಾರ, ನಿಮ್ಮ ಬಗ್ಗೆ ಹೇಳಿ", "type": "warmup", "tip": "ನಿಮ್ಮ ಹೆಸರು ಮತ್ತು ಅನುಭವ ಹೇಳಿ", "keywords": []},
            {"id": "f2", "text": "ನೀವು ಯಾವ ಕೆಲಸ ಮಾಡಿದ್ದೀರಿ", "type": "warmup", "tip": "ನಿಮ್ಮ ಕೆಲಸದ ಅನುಭವ ಹೇಳಿ", "keywords": []},
            {"id": "f3", "text": "ಕೆಲಸದಲ್ಲಿ ಸುರಕ್ಷತೆ ಏಕೆ ಮುಖ್ಯ", "type": "technical", "tip": "ಸುರಕ್ಷತಾ ಕ್ರಮಗಳ ಬಗ್ಗೆ ಹೇಳಿ", "keywords": ["ಸುರಕ್ಷತೆ"]},
            {"id": "f4", "text": "ನೀವು ಯಾವ ಉಪಕರಣಗಳನ್ನು ಬಳಸುತ್ತೀರಿ", "type": "technical", "tip": "ಉಪಕರಣಗಳ ಹೆಸರು ಹೇಳಿ", "keywords": []},
            {"id": "f5", "text": "ಸಮಸ್ಯೆ ಬಂದರೆ ಏನು ಮಾಡುತ್ತೀರಿ", "type": "technical", "tip": "ಸಮಸ್ಯೆ ಪರಿಹಾರ ಹೇಳಿ", "keywords": []},
            {"id": "f6", "text": "ಗುಣಮಟ್ಟದ ಕೆಲಸ ಹೇಗೆ ಮಾಡುತ್ತೀರಿ", "type": "technical", "tip": "ಗುಣಮಟ್ಟ ಖಚಿತಪಡಿಸುವ ವಿಧಾನ ಹೇಳಿ", "keywords": []},
            {"id": "f7", "text": "ಕಷ್ಟದ ಸಂದರ್ಭದಲ್ಲಿ ಹೇಗೆ ನಿಭಾಯಿಸುತ್ತೀರಿ", "type": "situational", "tip": "ನಿಮ್ಮ ಅನುಭವ ಹಂಚಿಕೊಳ್ಳಿ", "keywords": []}
        ],
        "hindi": [
            {"id": "f1", "text": "नमस्ते, अपने बारे में बताएं", "type": "warmup", "tip": "अपना नाम और अनुभव बताएं", "keywords": []},
            {"id": "f2", "text": "आपने क्या काम किया है", "type": "warmup", "tip": "अपना काम का अनुभव बताएं", "keywords": []},
            {"id": "f3", "text": "काम में सुरक्षा क्यों जरूरी है", "type": "technical", "tip": "सुरक्षा उपायों के बारे में बताएं", "keywords": ["सुरक्षा"]},
            {"id": "f4", "text": "आप कौन से औजार इस्तेमाल करते हैं", "type": "technical", "tip": "औजारों के नाम बताएं", "keywords": []},
            {"id": "f5", "text": "समस्या आने पर क्या करते हैं", "type": "technical", "tip": "समस्या समाधान बताएं", "keywords": []},
            {"id": "f6", "text": "अच्छा काम कैसे करते हैं", "type": "technical", "tip": "गुणवत्ता सुनिश्चित करने का तरीका बताएं", "keywords": []},
            {"id": "f7", "text": "मुश्किल हालात में कैसे संभालते हैं", "type": "situational", "tip": "अपना अनुभव साझा करें", "keywords": []}
        ],
        "english": [
            {"id": "f1", "text": "Hello, please tell us about yourself", "type": "warmup", "tip": "Tell your name and experience", "keywords": []},
            {"id": "f2", "text": "What work have you done", "type": "warmup", "tip": "Share your work experience", "keywords": []},
            {"id": "f3", "text": "Why is safety important in your work", "type": "technical", "tip": "Talk about safety measures", "keywords": ["safety"]},
            {"id": "f4", "text": "What tools do you use", "type": "technical", "tip": "Name the tools you use", "keywords": []},
            {"id": "f5", "text": "What do you do when a problem occurs", "type": "technical", "tip": "Explain your problem solving", "keywords": []},
            {"id": "f6", "text": "How do you ensure quality work", "type": "technical", "tip": "Explain your quality method", "keywords": []},
            {"id": "f7", "text": "How do you handle difficult situations", "type": "situational", "tip": "Share your experience", "keywords": []}
        ]
    }
    return questions.get(language, questions["english"])

# ─── Submit Response ──────────────────────────────────────────────────────────

@app.post("/api/interview/{session_id}/submit-response")
async def submit_response(
    session_id: str,
    background_tasks: BackgroundTasks,
    question_id: str = Form(...),
    question_text: str = Form(...),
    question_type: str = Form("technical"),
    video: UploadFile = File(...)
):
    """Save video response and queue background processing"""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Save video
    video_filename = f"{session_id}_{question_id}_{uuid.uuid4()}.webm"
    video_path = f"./uploads/videos/{video_filename}"
    with open(video_path, "wb") as f:
        f.write(await video.read())

    # Save response record
    response = db.save_response(
        session_id=session_id,
        question_id=question_id,
        question_text=question_text,
        question_type=question_type,
        video_url=f"/uploads/videos/{video_filename}"
    )

    # Queue background processing
    background_tasks.add_task(
        process_response_background,
        response["id"],
        video_path,
        {"id": question_id, "text": question_text, "keywords": []},
        session["language"],
        session.get("candidate_role", "electrician")
    )

    return {"success": True, "response_id": response["id"], "status": "processing"}

async def process_response_background(response_id: str, video_path: str,
                                       question: Dict, language: str, role: str):
    """Background task to process video response"""
    try:
        engine = get_assessment_engine()
        score = await engine.assess_response(video_path, question, language, role)

        db.update_response_scores(response_id, {
            "transcript": score.transcript,
            "ai_scores": {
                "technical": score.technical_score,
                "communication": score.communication_score,
                "confidence": score.confidence_score,
                "language": score.language_score,
                "overall": score.overall_score
            },
            "key_points_mentioned": score.keywords_found,
            "missing_points": score.keywords_missing,
            "red_flags": score.red_flags,
            "followup_question": score.follow_up_question
        })
    except Exception as e:
        print(f"Background processing error: {e}")
        db.update_response_scores(response_id, {
            "transcript": "",
            "ai_scores": {"technical": 5, "communication": 5, "confidence": 5, "language": 5, "overall": 5},
            "processing_status": "failed"
        })

# ─── Complete Interview ───────────────────────────────────────────────────────

@app.post("/api/interview/{session_id}/complete")
async def complete_interview(session_id: str):
    """Complete interview and generate final assessment"""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    candidate = db.get_candidate_by_id(session["candidate_id"])
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    # Wait briefly for background processing
    await asyncio.sleep(2)

    # Get all responses
    responses = db.get_session_responses(session_id)

    # Build question scores from responses
    from core.assessment_engine import QuestionScore
    question_scores = []
    for r in responses:
        ai_scores = r.get("ai_scores") or {}
        question_scores.append(QuestionScore(
            question_id=r.get("question_id", ""),
            question_text=r.get("question_text", ""),
            transcript=r.get("transcript", ""),
            technical_score=ai_scores.get("technical", 5.0),
            communication_score=ai_scores.get("communication", 5.0),
            confidence_score=ai_scores.get("confidence", 5.0),
            language_score=ai_scores.get("language", 5.0),
            relevance_score=ai_scores.get("overall", 5.0),
            overall_score=ai_scores.get("overall", 5.0),
            audio_features={},
            video_features={},
            keywords_found=r.get("key_points_mentioned") or [],
            keywords_missing=r.get("missing_points") or [],
            red_flags=r.get("red_flags") or [],
            ai_feedback="",
            follow_up_needed=False
        ))

    # If no responses yet, create mock scores for demo
    if not question_scores:
        for i in range(7):
            question_scores.append(QuestionScore(
                question_id=f"q{i+1}", question_text="", transcript="",
                technical_score=6.0, communication_score=6.0, confidence_score=6.0,
                language_score=7.0, relevance_score=6.0, overall_score=6.0,
                audio_features={}, video_features={}, keywords_found=[],
                keywords_missing=[], red_flags=[], ai_feedback="", follow_up_needed=False
            ))

    # Fraud analysis
    #fraud_engine = get_fraud_engine()
    video_features_list = [q.video_features for q in question_scores]
    audio_features_list = [q.audio_features for q in question_scores]
    fraud_result = fraud_engine.analyze_session(session_id, video_features_list, audio_features_list)

    # Generate final assessment
    engine = get_assessment_engine()
    final = await engine.generate_final_assessment(
        candidate_id=candidate["id"],
        session_id=session_id,
        question_scores=question_scores,
        fraud_result={"confidence": fraud_result.confidence, "fraud_flags": fraud_result.fraud_flags},
        candidate_data=candidate
    )

    # Save assessment
    from dataclasses import asdict
    assessment_data = {
        "overall_score": final.overall_score,
        "technical_score": final.technical_score,
        "communication_score": final.communication_score,
        "confidence_score": final.confidence_score,
        "language_score": final.language_score,
        "job_readiness_percentage": final.job_readiness_percentage,
        "category": final.category,
        "strengths": final.strengths,
        "weaknesses": final.weaknesses,
        "training_gaps": final.training_gaps,
        "training_recommendations": final.training_recommendations,
        "recommended_roles": final.recommended_roles,
        "ai_summary": final.ai_summary,
        "recruiter_recommendation": final.recruiter_recommendation,
        "fraud_score": final.fraud_score,
        "fraud_flags": final.fraud_flags
    }
    saved = db.save_assessment(candidate["id"], session_id, assessment_data)

    # Update session status
    db.update_session_status(session_id, "completed")

    return {
        "success": True,
        "category": final.category,
        "overall_score": final.overall_score,
        "job_readiness_percentage": final.job_readiness_percentage,
        "scores": {
            "technical": final.technical_score,
            "communication": final.communication_score,
            "confidence": final.confidence_score,
            "language": final.language_score
        },
        "strengths": final.strengths,
        "weaknesses": final.weaknesses,
        "recruiter_recommendation": final.recruiter_recommendation,
        "fraud_flags": final.fraud_flags
    }

# ─── Admin Auth ───────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: str
    password: str

@app.post("/api/auth/login")
async def admin_login(req: LoginRequest):
    admin = db.get_admin_by_email(req.email)
    if not admin:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    # Demo: plain text password check
    if admin.get("password_hash") != req.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(req.email)
    return {"token": token, "name": admin["name"], "role": admin["role"]}

# ─── Admin Candidates ─────────────────────────────────────────────────────────

@app.get("/api/admin/candidates")
async def get_candidates(
    district: Optional[str] = None,
    role: Optional[str] = None,
    language: Optional[str] = None,
    category: Optional[str] = None,
    status: Optional[str] = None,
    search: Optional[str] = None,
    email: str = Depends(verify_token)
):
    """Get all candidates with filters"""
    filters = {}
    if district: filters["district"] = district
    if role: filters["applied_role"] = role
    if language: filters["preferred_language"] = language

    candidates = db.get_all_candidates(filters)
    assessments = db.get_all_assessments({"category": category} if category else {})

    # Merge candidate + assessment data
    assessment_map = {a["candidate_id"]: a for a in assessments}
    result = []
    for c in candidates:
        assessment = assessment_map.get(c["id"], {})
        if category and assessment.get("category") != category:
            continue
        if status and assessment.get("final_status") != status:
            continue
        if search:
            search_lower = search.lower()
            if search_lower not in c.get("name", "").lower() and search_lower not in c.get("phone", ""):
                continue
        result.append({**c, "assessment": assessment})

    return {"candidates": result, "total": len(result)}

@app.get("/api/admin/candidates/{candidate_id}")
async def get_candidate_detail(candidate_id: str, email: str = Depends(verify_token)):
    """Get full candidate report"""
    candidate = db.get_candidate_by_id(candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    assessment = db.get_assessment_by_candidate(candidate_id)
    responses = []

    if assessment:
        session_id = assessment.get("session_id")
        if session_id:
            responses = db.get_session_responses(session_id)

    return {"candidate": candidate, "assessment": assessment, "responses": responses}

@app.put("/api/admin/candidates/{candidate_id}/status")
async def update_candidate_status(
    candidate_id: str,
    status: str = Form(...),
    notes: str = Form(""),
    email: str = Depends(verify_token)
):
    """Update candidate final status"""
    assessment = db.get_assessment_by_candidate(candidate_id)
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")

    db.update_assessment_status(assessment["id"], status, notes)
    return {"success": True, "status": status}

@app.get("/api/admin/stats")
async def get_stats(email: str = Depends(verify_token)):
    """Get dashboard statistics"""
    return db.get_dashboard_stats()

@app.get("/api/admin/analytics")
async def get_analytics(email: str = Depends(verify_token)):
    """Get analytics data for charts"""
    stats = db.get_dashboard_stats()
    assessments = db.get_all_assessments()
    candidates = db.get_all_candidates()

    # Score distribution
    score_ranges = {"0-4": 0, "4-6": 0, "6-8": 0, "8-10": 0}
    for a in assessments:
        score = a.get("overall_score", 0)
        if score < 4: score_ranges["0-4"] += 1
        elif score < 6: score_ranges["4-6"] += 1
        elif score < 8: score_ranges["6-8"] += 1
        else: score_ranges["8-10"] += 1

    # District distribution
    district_counts = {}
    for c in candidates:
        d = c.get("district", "Unknown")
        district_counts[d] = district_counts.get(d, 0) + 1

    # Role distribution
    role_counts = {}
    for c in candidates:
        r = c.get("applied_role", "Unknown")
        role_counts[r] = role_counts.get(r, 0) + 1

    return {
        "category_distribution": {
            "job_ready": stats["job_ready"],
            "needs_training": stats["needs_training"],
            "requires_verification": stats["requires_verification"],
            "low_quality": stats["low_quality"],
            "suspected_fraud": stats["suspected_fraud"]
        },
        "language_distribution": stats["by_language"],
        "score_distribution": score_ranges,
        "district_distribution": district_counts,
        "role_distribution": role_counts,
        "total_candidates": stats["total_candidates"]
    }

# ─── Candidate Dashboard ─────────────────────────────────────────────────────

@app.get("/api/candidate/{phone}/dashboard")
async def get_candidate_dashboard(phone: str):
    """Get candidate's assessment results and status"""
    candidate = db.get_candidate_by_phone(phone)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    
    assessment = db.get_assessment_by_candidate(candidate["id"])
    
    if not assessment:
        return {
            "candidate": candidate,
            "status": "pending",
            "message": "Your interview is being processed. Please check back later."
        }
    
    return {
        "candidate": candidate,
        "assessment": assessment,
        "status": "completed",
        "overall_score": assessment.get("overall_score"),
        "category": assessment.get("category"),
        "job_readiness_percentage": assessment.get("job_readiness_percentage"),
        "scores": {
            "technical": assessment.get("technical_score"),
            "communication": assessment.get("communication_score"),
            "confidence": assessment.get("confidence_score"),
            "language": assessment.get("language_score")
        },
        "strengths": assessment.get("strengths", []),
        "training_recommendations": assessment.get("training_recommendations", []),
        "final_status": assessment.get("final_status", "pending_review"),
        "recruiter_notes": assessment.get("recruiter_notes", "")
    }

# ─── Transcribe (for real-time) ───────────────────────────────────────────────

@app.post("/api/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: str = Form("kannada")
):
    """Transcribe audio using Whisper"""
    try:
        from core.speech_engine import get_speech_engine
        engine = get_speech_engine()

        temp_path = f"./uploads/{uuid.uuid4()}_{audio.filename}"
        with open(temp_path, "wb") as f:
            f.write(await audio.read())

        result = engine.transcribe(temp_path, language)
        os.remove(temp_path)

        return {"transcription": result.transcript, "language": result.detected_language, "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🚀 Starting AI SkillFit API v2.0")
    print(f"📊 Database: {'Supabase' if db.get_supabase() else 'In-Memory (Demo)'}")
    print("🌐 Server: http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=settings.debug)
    @app.get("/")
def home():
    return {"message": "AI SkillFit backend working"}
