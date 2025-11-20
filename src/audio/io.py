"""Audio input/output handling."""

import asyncio
from typing import Optional, Callable
import numpy as np
import pyaudio

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AudioInput:
    """Handle audio input from microphone."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        channels: int = 1,
        format: int = pyaudio.paFloat32
    ):
        """Initialize audio input.
        
        Args:
            sample_rate: Sample rate in Hz
            chunk_size: Number of samples per chunk
            channels: Number of audio channels
            format: Audio format
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = format
        
        self.pyaudio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self.is_recording = False
        self.callback: Optional[Callable] = None
        
        logger.info(f"Audio input initialized (rate={sample_rate}, chunk={chunk_size})")
    
    def start(self, callback: Callable[[np.ndarray], None]) -> None:
        """Start recording audio.
        
        Args:
            callback: Function to call with audio chunks
        """
        if self.is_recording:
            logger.warning("Already recording")
            return
        
        self.callback = callback
        
        def audio_callback(in_data, frame_count, time_info, status):
            if status:
                logger.warning(f"Audio input status: {status}")
            
            # Convert to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Call user callback
            if self.callback:
                try:
                    self.callback(audio_data.copy())
                except Exception as e:
                    logger.error(f"Error in audio callback: {e}")
            
            return (None, pyaudio.paContinue)
        
        try:
            self.stream = self.pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=audio_callback
            )
            
            self.stream.start_stream()
            self.is_recording = True
            logger.info("Started audio recording")
            
        except Exception as e:
            logger.error(f"Error starting audio input: {e}")
            raise
    
    def stop(self) -> None:
        """Stop recording audio."""
        if not self.is_recording:
            return
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        self.is_recording = False
        logger.info("Stopped audio recording")
    
    def close(self) -> None:
        """Close audio input and cleanup."""
        self.stop()
        self.pyaudio.terminate()
        logger.info("Audio input closed")


class AudioOutput:
    """Handle audio output to speakers."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        format: int = pyaudio.paFloat32
    ):
        """Initialize audio output.
        
        Args:
            sample_rate: Sample rate in Hz
            channels: Number of audio channels
            format: Audio format
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.format = format
        
        self.pyaudio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self.is_playing = False
        
        logger.info(f"Audio output initialized (rate={sample_rate})")
    
    def start(self) -> None:
        """Start audio output stream."""
        if self.is_playing:
            logger.warning("Already playing")
            return
        
        try:
            self.stream = self.pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                output=True
            )
            
            self.is_playing = True
            logger.info("Started audio playback")
            
        except Exception as e:
            logger.error(f"Error starting audio output: {e}")
            raise
    
    def play(self, audio_data: np.ndarray) -> None:
        """Play audio data.
        
        Args:
            audio_data: Audio samples (float32)
        """
        if not self.is_playing:
            self.start()
        
        try:
            # Ensure float32 format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Ensure proper range (-1 to 1)
            audio_data = np.clip(audio_data, -1.0, 1.0)
            
            # Write to stream
            self.stream.write(audio_data.tobytes())
            
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
    
    def stop(self) -> None:
        """Stop audio output."""
        if not self.is_playing:
            return
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        self.is_playing = False
        logger.info("Stopped audio playback")
    
    def close(self) -> None:
        """Close audio output and cleanup."""
        self.stop()
        self.pyaudio.terminate()
        logger.info("Audio output closed")


class AudioBuffer:
    """Thread-safe circular buffer for audio data."""
    
    def __init__(self, max_duration: float = 5.0, sample_rate: int = 16000):
        """Initialize audio buffer.
        
        Args:
            max_duration: Maximum buffer duration in seconds
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.buffer = np.zeros(self.max_samples, dtype=np.float32)
        self.write_pos = 0
        self.read_pos = 0
        self.lock = asyncio.Lock()
        
        logger.info(f"Audio buffer initialized (duration={max_duration}s, samples={self.max_samples})")
    
    async def write(self, data: np.ndarray) -> None:
        """Write audio data to buffer.
        
        Args:
            data: Audio samples
        """
        async with self.lock:
            data_len = len(data)
            
            # Handle wraparound
            if self.write_pos + data_len <= self.max_samples:
                self.buffer[self.write_pos:self.write_pos + data_len] = data
            else:
                # Split into two parts
                first_part = self.max_samples - self.write_pos
                self.buffer[self.write_pos:] = data[:first_part]
                self.buffer[:data_len - first_part] = data[first_part:]
            
            self.write_pos = (self.write_pos + data_len) % self.max_samples
    
    async def read(self, num_samples: int) -> Optional[np.ndarray]:
        """Read audio data from buffer.
        
        Args:
            num_samples: Number of samples to read
        
        Returns:
            Audio data or None if not enough data
        """
        async with self.lock:
            available = self.available_samples()
            if available < num_samples:
                return None
            
            # Handle wraparound
            if self.read_pos + num_samples <= self.max_samples:
                data = self.buffer[self.read_pos:self.read_pos + num_samples].copy()
            else:
                # Split into two parts
                first_part = self.max_samples - self.read_pos
                data = np.concatenate([
                    self.buffer[self.read_pos:],
                    self.buffer[:num_samples - first_part]
                ])
            
            self.read_pos = (self.read_pos + num_samples) % self.max_samples
            return data
    
    def available_samples(self) -> int:
        """Get number of available samples.
        
        Returns:
            Number of samples available
        """
        if self.write_pos >= self.read_pos:
            return self.write_pos - self.read_pos
        else:
            return self.max_samples - self.read_pos + self.write_pos
    
    async def clear(self) -> None:
        """Clear the buffer."""
        async with self.lock:
            self.buffer.fill(0)
            self.write_pos = 0
            self.read_pos = 0
