"""Audio synchronization utilities."""

import asyncio
import time
from typing import Optional
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AudioSynchronizer:
    """Synchronize audio playback with incoming packets."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        target_buffer_ms: int = 200,
        max_drift_ms: int = 100
    ):
        """Initialize audio synchronizer.
        
        Args:
            sample_rate: Sample rate in Hz
            target_buffer_ms: Target buffer size in milliseconds
            max_drift_ms: Maximum allowed drift in milliseconds
        """
        self.sample_rate = sample_rate
        self.target_buffer_samples = int(target_buffer_ms * sample_rate / 1000)
        self.max_drift_samples = int(max_drift_ms * sample_rate / 1000)
        
        self.last_play_time: Optional[float] = None
        self.samples_played = 0
        self.samples_received = 0
        
        logger.info(f"Audio synchronizer initialized (target={target_buffer_ms}ms)")
    
    def on_audio_received(self, num_samples: int) -> None:
        """Called when audio samples are received.
        
        Args:
            num_samples: Number of samples received
        """
        self.samples_received += num_samples
    
    def on_audio_played(self, num_samples: int) -> None:
        """Called when audio samples are played.
        
        Args:
            num_samples: Number of samples played
        """
        self.samples_played += num_samples
        self.last_play_time = time.time()
    
    def get_buffer_level(self) -> int:
        """Get current buffer level in samples.
        
        Returns:
            Number of samples in buffer
        """
        return max(0, self.samples_received - self.samples_played)
    
    def get_buffer_level_ms(self) -> float:
        """Get current buffer level in milliseconds.
        
        Returns:
            Buffer level in ms
        """
        return (self.get_buffer_level() / self.sample_rate) * 1000
    
    def should_skip_frame(self) -> bool:
        """Check if we should skip a frame due to buffer overflow.
        
        Returns:
            True if should skip
        """
        buffer_level = self.get_buffer_level()
        return buffer_level > (self.target_buffer_samples + self.max_drift_samples)
    
    def should_repeat_frame(self) -> bool:
        """Check if we should repeat a frame due to buffer underflow.
        
        Returns:
            True if should repeat
        """
        buffer_level = self.get_buffer_level()
        return buffer_level < (self.target_buffer_samples - self.max_drift_samples)
    
    def get_playback_rate_adjustment(self) -> float:
        """Get playback rate adjustment factor.
        
        Returns:
            Rate adjustment (1.0 = normal, >1.0 = faster, <1.0 = slower)
        """
        buffer_level = self.get_buffer_level()
        drift = buffer_level - self.target_buffer_samples
        
        # Adjust playback rate based on drift
        if abs(drift) < self.max_drift_samples / 4:
            return 1.0  # Normal playback
        elif drift > 0:
            # Buffer too full, speed up slightly
            return 1.0 + min(0.05, drift / (self.max_drift_samples * 2))
        else:
            # Buffer too empty, slow down slightly
            return 1.0 - min(0.05, abs(drift) / (self.max_drift_samples * 2))
    
    def reset(self) -> None:
        """Reset synchronization state."""
        self.samples_played = 0
        self.samples_received = 0
        self.last_play_time = None
        logger.info("Audio synchronizer reset")


class JitterBuffer:
    """Adaptive jitter buffer for audio packets."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        initial_delay_ms: int = 100,
        max_delay_ms: int = 300
    ):
        """Initialize jitter buffer.
        
        Args:
            sample_rate: Sample rate in Hz
            initial_delay_ms: Initial buffering delay
            max_delay_ms: Maximum buffering delay
        """
        self.sample_rate = sample_rate
        self.initial_delay_samples = int(initial_delay_ms * sample_rate / 1000)
        self.max_delay_samples = int(max_delay_ms * sample_rate / 1000)
        
        self.buffer: dict[int, np.ndarray] = {}
        self.next_sequence = 0
        self.buffering = True
        self.lock = asyncio.Lock()
        
        # Statistics
        self.packets_received = 0
        self.packets_lost = 0
        self.late_packets = 0
        
        logger.info(f"Jitter buffer initialized (initial={initial_delay_ms}ms, max={max_delay_ms}ms)")
    
    async def add_packet(self, sequence: int, audio_data: np.ndarray) -> None:
        """Add packet to jitter buffer.
        
        Args:
            sequence: Packet sequence number
            audio_data: Audio samples
        """
        async with self.lock:
            self.packets_received += 1
            
            # Check if packet is late
            if sequence < self.next_sequence:
                self.late_packets += 1
                logger.debug(f"Late packet: seq={sequence} (expected={self.next_sequence})")
                return
            
            # Add to buffer
            self.buffer[sequence] = audio_data
            
            # Check buffer size
            if self.buffering and len(self.buffer) * len(audio_data) >= self.initial_delay_samples:
                self.buffering = False
                logger.info("Jitter buffer ready for playback")
    
    async def get_next_packet(self) -> Optional[np.ndarray]:
        """Get next packet from buffer.
        
        Returns:
            Audio data or None if not available
        """
        async with self.lock:
            # Still buffering
            if self.buffering:
                return None
            
            # Check if next packet is available
            if self.next_sequence in self.buffer:
                audio_data = self.buffer.pop(self.next_sequence)
                self.next_sequence += 1
                return audio_data
            else:
                # Packet lost
                self.packets_lost += 1
                self.next_sequence += 1
                logger.debug(f"Packet lost: seq={self.next_sequence - 1}")
                
                # Return silence
                # Estimate packet size from buffer
                if self.buffer:
                    packet_size = len(next(iter(self.buffer.values())))
                else:
                    packet_size = 1024  # Default
                
                return np.zeros(packet_size, dtype=np.float32)
    
    def get_statistics(self) -> dict:
        """Get jitter buffer statistics.
        
        Returns:
            Statistics dictionary
        """
        loss_rate = (self.packets_lost / max(1, self.packets_received)) * 100
        late_rate = (self.late_packets / max(1, self.packets_received)) * 100
        
        return {
            "packets_received": self.packets_received,
            "packets_lost": self.packets_lost,
            "late_packets": self.late_packets,
            "loss_rate_percent": loss_rate,
            "late_rate_percent": late_rate,
            "buffer_size": len(self.buffer),
            "buffering": self.buffering,
        }
    
    async def clear(self) -> None:
        """Clear the buffer."""
        async with self.lock:
            self.buffer.clear()
            self.buffering = True
            logger.info("Jitter buffer cleared")
