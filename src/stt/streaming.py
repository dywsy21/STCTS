"""Streaming STT handler."""

import queue
import threading
from typing import Callable, Iterator, Optional

import numpy as np

from src.stt.base import BaseSTT, STTResult
from src.stt.vad import VAD
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StreamingSTTHandler:
    """Handle streaming speech-to-text transcription."""
    
    def __init__(
        self,
        stt_model: BaseSTT,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 400,
        use_vad: bool = True,
        vad_mode: int = 3,
        on_result: Optional[Callable[[STTResult], None]] = None,
    ):
        """Initialize streaming STT handler.
        
        Args:
            stt_model: STT model instance
            sample_rate: Audio sample rate
            chunk_duration_ms: Duration of each audio chunk in ms
            use_vad: Whether to use VAD
            vad_mode: VAD aggressiveness (0-3)
            on_result: Callback function for transcription results
        """
        self.stt_model = stt_model
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self.use_vad = use_vad
        self.on_result = on_result
        
        # VAD
        self.vad = VAD(sample_rate=sample_rate, mode=vad_mode) if use_vad else None
        
        # Audio buffer
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        
        # Processing thread
        self.processing_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.audio_queue: queue.Queue = queue.Queue()
        
        # State
        self.is_running = False
        self.total_frames_processed = 0
        
        logger.info("Streaming STT handler initialized")
    
    def start(self) -> None:
        """Start the streaming handler."""
        if self.is_running:
            logger.warning("Handler already running")
            return
        
        # Ensure model is loaded
        if not self.stt_model.is_loaded:
            self.stt_model.load_model()
        
        self.stop_event.clear()
        self.is_running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        logger.info("Streaming handler started")
    
    def stop(self) -> None:
        """Stop the streaming handler."""
        if not self.is_running:
            return
        
        self.is_running = False
        self.stop_event.set()
        
        # Wait for thread to finish
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        logger.info("Streaming handler stopped")
    
    def add_audio(self, audio_chunk: np.ndarray) -> None:
        """Add audio chunk to processing queue.
        
        Args:
            audio_chunk: Audio data as numpy array
        """
        if not self.is_running:
            logger.warning("Handler not running, ignoring audio")
            return
        
        # Apply VAD if enabled
        if self.vad:
            # Ensure int16 for VAD
            if audio_chunk.dtype != np.int16:
                if audio_chunk.dtype in [np.float32, np.float64]:
                    audio_chunk_int16 = (audio_chunk * 32767).astype(np.int16)
                else:
                    audio_chunk_int16 = audio_chunk.astype(np.int16)
            else:
                audio_chunk_int16 = audio_chunk
            
            # Check for speech
            has_speech = False
            frame_size = self.vad.frame_size
            
            for i in range(0, len(audio_chunk_int16), frame_size):
                frame = audio_chunk_int16[i:i+frame_size]
                if len(frame) == frame_size and self.vad.is_speech(frame):
                    has_speech = True
                    break
            
            if not has_speech:
                # No speech, skip this chunk
                return
        
        # Add to queue
        self.audio_queue.put(audio_chunk)
    
    def _processing_loop(self) -> None:
        """Main processing loop (runs in separate thread)."""
        logger.info("Processing loop started")
        
        while not self.stop_event.is_set():
            try:
                # Get audio chunk with timeout
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Add to buffer
                with self.buffer_lock:
                    self.audio_buffer.append(audio_chunk)
                    
                    # Calculate total duration
                    total_samples = sum(len(chunk) for chunk in self.audio_buffer)
                    duration_seconds = total_samples / self.sample_rate
                    
                    # Process when buffer reaches target duration
                    target_duration = self.chunk_duration_ms / 1000.0
                    if duration_seconds >= target_duration:
                        # Combine buffer
                        combined_audio = np.concatenate(self.audio_buffer)
                        self.audio_buffer = []
                        
                        # Transcribe
                        try:
                            result = self.stt_model.transcribe(combined_audio, self.sample_rate)
                            
                            if result.text:
                                self.total_frames_processed += 1
                                logger.debug(f"Transcribed: {result.text}")
                                
                                # Call callback
                                if self.on_result:
                                    self.on_result(result)
                        
                        except Exception as e:
                            logger.error(f"Error transcribing audio: {e}")
            
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
        
        logger.info("Processing loop stopped")
    
    def get_stats(self) -> dict:
        """Get handler statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "is_running": self.is_running,
            "total_frames_processed": self.total_frames_processed,
            "queue_size": self.audio_queue.qsize(),
            "buffer_size": len(self.audio_buffer),
        }
