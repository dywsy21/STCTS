"""Faster-Whisper STT implementation."""

from typing import Iterator, Optional

import numpy as np
from faster_whisper import WhisperModel

from src.stt.base import BaseSTT, STTResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FasterWhisperSTT(BaseSTT):
    """Faster-Whisper implementation of STT."""
    
    def __init__(
        self,
        model_size: str = "distil-large-v3",
        device: str = "auto",
        compute_type: str = "default",
        language: Optional[str] = None,
    ):
        """Initialize Faster-Whisper STT.
        
        Args:
            model_size: Model size (tiny, base, small, medium, large, distil-large-v3)
            device: Device to run on (cpu, cuda, auto)
            compute_type: Computation type (int8, float16, float32, default)
            language: Language code (optional, for better performance)
        """
        super().__init__(model_size, language)
        
        # Auto-detect device if set to auto
        if device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Auto-configure compute type based on device
        if compute_type == "default":
            if self.device == "cuda":
                self.compute_type = "float16"
            else:
                self.compute_type = "int8"
        else:
            self.compute_type = compute_type
            
        self.model: Optional[WhisperModel] = None
    
    def load_model(self) -> None:
        """Load the Faster-Whisper model."""
        if self.is_loaded:
            logger.info("Model already loaded")
            return
        
        logger.info(
            f"Loading Faster-Whisper model: {self.model_name} "
            f"(device={self.device}, compute_type={self.compute_type})"
        )
        
        self.model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )
        
        self.is_loaded = True
        logger.info("Model loaded successfully")
    
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> STTResult:
        """Transcribe audio to text.
        
        Args:
            audio: Audio data as numpy array (float32, range -1 to 1)
            sample_rate: Audio sample rate
        
        Returns:
            Transcription result
        """
        if not self.is_loaded:
            self.load_model()
        
        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Transcribe with strict VAD settings to filter noise
        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            vad_filter=True,
            vad_parameters=dict(
                threshold=0.6,  # Higher threshold to reduce noise sensitivity
                min_speech_duration_ms=500,  # Require at least 500ms of speech
                max_speech_duration_s=5000000,
                min_silence_duration_ms=500,  # Require 500ms silence before segmenting
                speech_pad_ms=400,  # Padding around speech segments
            ),
        )
        
        # Collect all segments
        all_text = []
        start_time = float('inf')
        end_time = 0.0
        total_confidence = 0.0
        num_segments = 0
        
        for segment in segments:
            all_text.append(segment.text)
            start_time = min(start_time, segment.start)
            end_time = max(end_time, segment.end)
            
            # Average log probability as confidence
            if hasattr(segment, 'avg_logprob'):
                # Convert log prob to probability (rough approximation)
                confidence = np.exp(segment.avg_logprob)
                total_confidence += confidence
                num_segments += 1
        
        text = " ".join(all_text).strip()
        avg_confidence = total_confidence / num_segments if num_segments > 0 else 0.0
        
        if start_time == float('inf'):
            start_time = 0.0
        
        # Filter out very short transcriptions (likely noise or incomplete)
        # Require at least 2 characters for valid transcription
        if len(text) < 2:
            return STTResult(
                text="",
                confidence=0.0,
                start_time=0.0,
                end_time=0.0,
            )
        
        return STTResult(
            text=text,
            confidence=float(avg_confidence),
            start_time=start_time,
            end_time=end_time,
            language=info.language if hasattr(info, 'language') else self.language,
        )
    
    def transcribe_stream(
        self,
        audio_stream: Iterator[np.ndarray],
        sample_rate: int = 16000,
    ) -> Iterator[STTResult]:
        """Transcribe streaming audio.
        
        Args:
            audio_stream: Iterator of audio chunks
            sample_rate: Audio sample rate
        
        Yields:
            Transcription results
        """
        if not self.is_loaded:
            self.load_model()
        
        # Buffer for accumulating audio
        audio_buffer = []
        buffer_duration = 0.0
        chunk_duration = 0.4  # 400ms chunks
        
        for audio_chunk in audio_stream:
            # Ensure float32
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)
            
            audio_buffer.append(audio_chunk)
            buffer_duration += len(audio_chunk) / sample_rate
            
            # Process when buffer reaches target duration
            if buffer_duration >= chunk_duration:
                combined_audio = np.concatenate(audio_buffer)
                
                try:
                    result = self.transcribe(combined_audio, sample_rate)
                    if result.text:  # Only yield if we got text
                        yield result
                except Exception as e:
                    logger.error(f"Error transcribing chunk: {e}")
                
                # Reset buffer
                audio_buffer = []
                buffer_duration = 0.0
        
        # Process remaining audio
        if audio_buffer:
            combined_audio = np.concatenate(audio_buffer)
            try:
                result = self.transcribe(combined_audio, sample_rate)
                if result.text:
                    yield result
            except Exception as e:
                logger.error(f"Error transcribing final chunk: {e}")
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        self.model = None
        self.is_loaded = False
        logger.info("Model unloaded")
