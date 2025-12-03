"""Voice Activity Detection module."""

import numpy as np
import webrtcvad

from src.utils.logger import get_logger

logger = get_logger(__name__)


class VAD:
    """Voice Activity Detection using WebRTC VAD."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration: int = 30,  # ms
        mode: int = 3,  # Aggressiveness (0-3)
    ):
        """Initialize VAD.
        
        Args:
            sample_rate: Audio sample rate (8000, 16000, 32000, 48000)
            frame_duration: Frame duration in ms (10, 20, 30)
            mode: Aggressiveness mode (0=least, 3=most aggressive)
        """
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.mode = mode
        
        # Validate parameters
        if sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError(f"Invalid sample rate: {sample_rate}")
        if frame_duration not in [10, 20, 30]:
            raise ValueError(f"Invalid frame duration: {frame_duration}")
        if mode not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid mode: {mode}")
        
        self.vad = webrtcvad.Vad(mode)
        self.frame_size = int(sample_rate * frame_duration / 1000)
        
        logger.info(
            f"VAD initialized: sample_rate={sample_rate}, "
            f"frame_duration={frame_duration}ms, mode={mode}"
        )
    
    def is_speech(self, audio: np.ndarray) -> bool:
        """Detect if audio frame contains speech.
        
        Args:
            audio: Audio frame as numpy array (int16)
        
        Returns:
            True if speech detected, False otherwise
        """
        # Ensure correct length
        if len(audio) != self.frame_size:
            # Pad or truncate
            if len(audio) < self.frame_size:
                audio = np.pad(audio, (0, self.frame_size - len(audio)))
            else:
                audio = audio[:self.frame_size]
        
        # Convert to int16 if needed
        if audio.dtype != np.int16:
            if audio.dtype == np.float32 or audio.dtype == np.float64:
                # Assume range -1 to 1
                audio = (audio * 32767).astype(np.int16)
            else:
                audio = audio.astype(np.int16)
        
        # Convert to bytes
        audio_bytes = audio.tobytes()
        
        return self.vad.is_speech(audio_bytes, self.sample_rate)
    
    def filter_audio(
        self,
        audio: np.ndarray,
        padding_duration: float = 0.3,  # seconds
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Filter audio to keep only speech segments.
        
        Args:
            audio: Full audio as numpy array (int16)
            padding_duration: Duration to pad around speech (seconds)
        
        Returns:
            Tuple of (filtered audio, list of (start, end) indices)
        """
        num_padding_frames = int(padding_duration * self.sample_rate / self.frame_size)
        
        # Split audio into frames
        num_frames = len(audio) // self.frame_size
        frames = []
        speech_flags = []
        
        for i in range(num_frames):
            start = i * self.frame_size
            end = start + self.frame_size
            frame = audio[start:end]
            
            frames.append(frame)
            speech_flags.append(self.is_speech(frame))
        
        # Add padding around speech segments
        padded_flags = speech_flags.copy()
        for i, is_speech in enumerate(speech_flags):
            if is_speech:
                # Add padding before
                for j in range(max(0, i - num_padding_frames), i):
                    padded_flags[j] = True
                # Add padding after
                for j in range(i + 1, min(len(frames), i + num_padding_frames + 1)):
                    padded_flags[j] = True
        
        # Extract speech segments
        segments = []
        current_segment = []
        segment_indices = []
        
        for i, (frame, is_speech) in enumerate(zip(frames, padded_flags)):
            if is_speech:
                if not current_segment:  # Start of new segment
                    segment_start = i * self.frame_size
                current_segment.append(frame)
            else:
                if current_segment:  # End of segment
                    segment_end = i * self.frame_size
                    segments.append(np.concatenate(current_segment))
                    segment_indices.append((segment_start, segment_end))
                    current_segment = []
        
        # Handle last segment
        if current_segment:
            segment_end = len(audio)
            segments.append(np.concatenate(current_segment))
            segment_indices.append((segment_start, segment_end))
        
        # Concatenate all speech segments
        if segments:
            filtered_audio = np.concatenate(segments)
        else:
            filtered_audio = np.array([], dtype=audio.dtype)
        
        return filtered_audio, segment_indices
