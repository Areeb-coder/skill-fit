"""
Audio Analyzer - Analyzes speech quality using librosa
Evaluates: clarity, confidence, speech rate, pauses, energy
"""
import librosa
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class AudioFeatures:
    """Audio analysis results"""
    duration_seconds: float
    speech_rate_wpm: float  # Words per minute
    avg_energy: float
    energy_variance: float
    avg_pitch_hz: float
    pitch_stability: float
    pause_count: int
    avg_pause_duration: float
    long_pause_count: int  # Pauses > 2 seconds
    filler_ratio: float  # Ratio of filler words
    snr_db: float  # Signal-to-noise ratio
    audio_quality: str  # "good", "acceptable", "poor"
    speaking_ratio: float  # Ratio of speech to silence
    clarity_score: float  # 0-10
    confidence_score: float  # 0-10
    overall_audio_score: float  # 0-10

class AudioAnalyzer:
    def __init__(self):
        self.sample_rate = 16000
        
        # Filler words in all 3 languages
        self.filler_words = {
            'kannada': ['ಅಂದರೆ', 'ಏನು', 'ಹಾಗೆ', 'ಹೀಗೆ', 'ಅಲ್ವಾ'],
            'hindi': ['उम', 'आह', 'मतलब', 'वो', 'है ना', 'तो'],
            'english': ['um', 'uh', 'er', 'hmm', 'like', 'you know', 'so']
        }
    
    def analyze(self, audio_path: str, transcript: str = "", language: str = "kannada") -> AudioFeatures:
        """
        Analyze audio file and return comprehensive features
        
        Args:
            audio_path: Path to audio file
            transcript: Transcribed text (optional, for filler word detection)
            language: Language of speech
        
        Returns:
            AudioFeatures with all metrics
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        duration = len(y) / sr
        
        # Calculate all features
        speech_rate = self._calculate_speech_rate(y, sr, transcript)
        energy_metrics = self._calculate_energy(y)
        pitch_metrics = self._calculate_pitch(y, sr)
        pause_metrics = self._calculate_pauses(y, sr)
        filler_ratio = self._calculate_filler_ratio(transcript, language)
        snr = self._calculate_snr(y)
        speaking_ratio = self._calculate_speaking_ratio(y, sr)
        
        # Determine audio quality
        audio_quality = self._determine_audio_quality(snr, energy_metrics['avg_energy'])
        
        # Calculate scores
        clarity_score = self._calculate_clarity_score(
            speech_rate, energy_metrics, pause_metrics, snr
        )
        confidence_score = self._calculate_confidence_score(
            energy_metrics, pitch_metrics, pause_metrics, filler_ratio
        )
        overall_score = (clarity_score + confidence_score) / 2
        
        return AudioFeatures(
            duration_seconds=duration,
            speech_rate_wpm=speech_rate,
            avg_energy=energy_metrics['avg_energy'],
            energy_variance=energy_metrics['energy_variance'],
            avg_pitch_hz=pitch_metrics['avg_pitch'],
            pitch_stability=pitch_metrics['pitch_stability'],
            pause_count=pause_metrics['pause_count'],
            avg_pause_duration=pause_metrics['avg_pause_duration'],
            long_pause_count=pause_metrics['long_pause_count'],
            filler_ratio=filler_ratio,
            snr_db=snr,
            audio_quality=audio_quality,
            speaking_ratio=speaking_ratio,
            clarity_score=clarity_score,
            confidence_score=confidence_score,
            overall_audio_score=overall_score
        )
    
    def _calculate_speech_rate(self, y: np.ndarray, sr: int, transcript: str) -> float:
        """Calculate words per minute"""
        duration_minutes = len(y) / sr / 60
        if transcript and duration_minutes > 0:
            word_count = len(transcript.split())
            return word_count / duration_minutes
        return 0.0
    
    def _calculate_energy(self, y: np.ndarray) -> dict:
        """Calculate energy metrics"""
        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        avg_energy = float(np.mean(rms))
        energy_variance = float(np.var(rms))
        
        return {
            'avg_energy': avg_energy,
            'energy_variance': energy_variance
        }
    
    def _calculate_pitch(self, y: np.ndarray, sr: int) -> dict:
        """Calculate pitch metrics"""
        # Extract pitch using piptrack
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        
        # Get pitch values where magnitude is highest
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:  # Valid pitch
                pitch_values.append(pitch)
        
        if len(pitch_values) > 0:
            avg_pitch = float(np.mean(pitch_values))
            pitch_std = float(np.std(pitch_values))
            pitch_stability = 1.0 / (1.0 + pitch_std / avg_pitch) if avg_pitch > 0 else 0.0
        else:
            avg_pitch = 0.0
            pitch_stability = 0.0
        
        return {
            'avg_pitch': avg_pitch,
            'pitch_stability': pitch_stability
        }
    
    def _calculate_pauses(self, y: np.ndarray, sr: int) -> dict:
        """Calculate pause metrics"""
        # Detect non-silent intervals
        intervals = librosa.effects.split(y, top_db=30)
        
        # Calculate pauses between intervals
        pauses = []
        for i in range(len(intervals) - 1):
            pause_start = intervals[i][1]
            pause_end = intervals[i + 1][0]
            pause_duration = (pause_end - pause_start) / sr
            if pause_duration > 0.1:  # Minimum 0.1s to count as pause
                pauses.append(pause_duration)
        
        pause_count = len(pauses)
        avg_pause_duration = float(np.mean(pauses)) if pauses else 0.0
        long_pause_count = sum(1 for p in pauses if p > 2.0)
        
        return {
            'pause_count': pause_count,
            'avg_pause_duration': avg_pause_duration,
            'long_pause_count': long_pause_count
        }
    
    def _calculate_filler_ratio(self, transcript: str, language: str) -> float:
        """Calculate ratio of filler words"""
        if not transcript:
            return 0.0
        
        words = transcript.lower().split()
        if len(words) == 0:
            return 0.0
        
        filler_words = self.filler_words.get(language, self.filler_words['english'])
        filler_count = sum(1 for word in words if word in filler_words)
        
        return filler_count / len(words)
    
    def _calculate_snr(self, y: np.ndarray) -> float:
        """Calculate signal-to-noise ratio"""
        # Simple SNR estimation
        # Signal: high energy frames, Noise: low energy frames
        rms = librosa.feature.rms(y=y)[0]
        threshold = np.percentile(rms, 50)
        
        signal = rms[rms > threshold]
        noise = rms[rms <= threshold]
        
        if len(noise) > 0 and np.mean(noise) > 0:
            snr = 20 * np.log10(np.mean(signal) / np.mean(noise))
            return float(snr)
        return 20.0  # Default good SNR
    
    def _calculate_speaking_ratio(self, y: np.ndarray, sr: int) -> float:
        """Calculate ratio of speech to total duration"""
        intervals = librosa.effects.split(y, top_db=30)
        speech_duration = sum((end - start) / sr for start, end in intervals)
        total_duration = len(y) / sr
        
        return speech_duration / total_duration if total_duration > 0 else 0.0
    
    def _determine_audio_quality(self, snr: float, avg_energy: float) -> str:
        """Determine overall audio quality"""
        if snr > 15 and avg_energy > 0.01:
            return "good"
        elif snr > 10 and avg_energy > 0.005:
            return "acceptable"
        else:
            return "poor"
    
    def _calculate_clarity_score(self, speech_rate: float, energy_metrics: dict, 
                                  pause_metrics: dict, snr: float) -> float:
        """Calculate clarity score (0-10)"""
        score = 10.0
        
        # Penalize very slow or very fast speech
        if speech_rate < 80 or speech_rate > 200:
            score -= 2.0
        
        # Penalize low energy
        if energy_metrics['avg_energy'] < 0.005:
            score -= 2.0
        
        # Penalize too many long pauses
        if pause_metrics['long_pause_count'] > 3:
            score -= 1.5
        
        # Penalize poor SNR
        if snr < 10:
            score -= 2.0
        elif snr < 15:
            score -= 1.0
        
        return max(0.0, min(10.0, score))
    
    def _calculate_confidence_score(self, energy_metrics: dict, pitch_metrics: dict,
                                     pause_metrics: dict, filler_ratio: float) -> float:
        """Calculate confidence score (0-10)"""
        score = 10.0
        
        # Penalize low energy (hesitant speech)
        if energy_metrics['avg_energy'] < 0.01:
            score -= 2.0
        
        # Penalize high energy variance (inconsistent)
        if energy_metrics['energy_variance'] > 0.001:
            score -= 1.0
        
        # Penalize unstable pitch (nervous)
        if pitch_metrics['pitch_stability'] < 0.7:
            score -= 1.5
        
        # Penalize too many pauses
        if pause_metrics['pause_count'] > 10:
            score -= 1.0
        
        # Penalize long pauses
        if pause_metrics['long_pause_count'] > 2:
            score -= 1.5
        
        # Penalize high filler ratio
        if filler_ratio > 0.1:
            score -= 2.0
        elif filler_ratio > 0.05:
            score -= 1.0
        
        return max(0.0, min(10.0, score))

# Singleton instance
_audio_analyzer = None

def get_audio_analyzer() -> AudioAnalyzer:
    """Get singleton audio analyzer instance"""
    global _audio_analyzer
    if _audio_analyzer is None:
        _audio_analyzer = AudioAnalyzer()
    return _audio_analyzer
