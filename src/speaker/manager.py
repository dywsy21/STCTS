"""Speaker management system."""

from pathlib import Path
from typing import Optional, Dict
import numpy as np
import time

from src.speaker.embedding import SpeakerEmbedding
from src.speaker.profile import TimbreProfile
from src.speaker.change_detection import detect_speaker_change
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SpeakerManager:
    """Manage speaker profiles and detect changes."""
    
    def __init__(
        self,
        model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        change_threshold: float = 0.3,
        profile_dir: Optional[Path] = None,
        device: str = "auto"
    ):
        """Initialize speaker manager.
        
        Args:
            model_name: SpeechBrain model name for embeddings
            change_threshold: Threshold for speaker change detection
            profile_dir: Directory to store speaker profiles
            device: Device to run model on (cpu, cuda, auto)
        """
        self.embedding_extractor = SpeakerEmbedding(model_name=model_name, device=device)
        self.change_threshold = change_threshold
        self.profile_dir = profile_dir or Path("./profiles")
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        
        # Current speaker state
        self.current_profile: Optional[TimbreProfile] = None
        self.profiles: Dict[str, TimbreProfile] = {}
        
        logger.info(f"Speaker manager initialized (threshold={change_threshold})")
    
    def extract_embedding(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> np.ndarray:
        """Extract speaker embedding from audio.
        
        Args:
            audio: Audio data
            sample_rate: Sample rate
        
        Returns:
            Speaker embedding vector
        """
        return self.embedding_extractor.extract(audio, sample_rate)

    def identify_from_embedding(self, embedding: np.ndarray) -> Optional[str]:
        """Identify speaker from embedding.
        
        Args:
            embedding: Speaker embedding
            
        Returns:
            Speaker ID if match found, None otherwise
        """
        if not self.profiles:
            return None
            
        best_match = None
        best_distance = float('inf')
        
        from scipy.spatial.distance import cosine
        
        for speaker_id, profile in self.profiles.items():
            distance = cosine(embedding, profile.embedding)
            
            if distance < best_distance:
                best_distance = distance
                best_match = speaker_id
        
        if best_distance < self.change_threshold:
            return best_match
        return None
    
    def create_profile(
        self,
        speaker_id: str,
        audio: Optional[np.ndarray] = None,
        embedding: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
        avg_pitch: float = 0.0,
        avg_speaking_rate: float = 0.0
    ) -> TimbreProfile:
        """Create a new speaker profile.
        
        Args:
            speaker_id: Unique speaker identifier
            audio: Audio sample from speaker (optional if embedding provided)
            embedding: Speaker embedding (optional if audio provided)
            sample_rate: Sample rate
            avg_pitch: Average pitch (F0) in Hz
            avg_speaking_rate: Average speaking rate in words/min
        
        Returns:
            New timbre profile
        """
        if embedding is None:
            if audio is None:
                raise ValueError("Either audio or embedding must be provided")
            # Extract embedding
            embedding = self.extract_embedding(audio, sample_rate)
        
        # Create profile
        profile = TimbreProfile(
            speaker_id=speaker_id,
            embedding=embedding,
            avg_pitch=avg_pitch,
            avg_speaking_rate=avg_speaking_rate,
            timestamp=time.time()
        )
        
        # Store in memory
        self.profiles[speaker_id] = profile
        self.current_profile = profile
        
        # Save to disk
        profile_path = self.profile_dir / f"{speaker_id}.json"
        profile.save(profile_path)
        
        logger.info(f"Created profile for speaker: {speaker_id}")
        return profile
    
    def load_profile(self, speaker_id: str) -> Optional[TimbreProfile]:
        """Load a speaker profile from disk.
        
        Args:
            speaker_id: Speaker identifier
        
        Returns:
            Timbre profile or None if not found
        """
        profile_path = self.profile_dir / f"{speaker_id}.json"
        
        if not profile_path.exists():
            logger.warning(f"Profile not found: {speaker_id}")
            return None
        
        try:
            profile = TimbreProfile.load(profile_path)
            self.profiles[speaker_id] = profile
            return profile
        except Exception as e:
            logger.error(f"Error loading profile {speaker_id}: {e}")
            return None
    
    def update_current_profile(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        avg_pitch: Optional[float] = None,
        avg_speaking_rate: Optional[float] = None
    ) -> Optional[TimbreProfile]:
        """Update the current speaker profile.
        
        Args:
            audio: New audio sample
            sample_rate: Sample rate
            avg_pitch: Updated average pitch
            avg_speaking_rate: Updated speaking rate
        
        Returns:
            Updated profile or None if no current profile
        """
        if self.current_profile is None:
            logger.warning("No current profile to update")
            return None
        
        # Extract new embedding
        new_embedding = self.extract_embedding(audio, sample_rate)
        
        # Blend with existing embedding (exponential moving average)
        alpha = 0.7  # Weight for new sample
        blended_embedding = (
            alpha * new_embedding +
            (1 - alpha) * self.current_profile.embedding
        )
        
        # Update profile
        self.current_profile.embedding = blended_embedding
        if avg_pitch is not None:
            self.current_profile.avg_pitch = avg_pitch
        if avg_speaking_rate is not None:
            self.current_profile.avg_speaking_rate = avg_speaking_rate
        self.current_profile.timestamp = time.time()
        
        # Save updated profile
        profile_path = self.profile_dir / f"{self.current_profile.speaker_id}.json"
        self.current_profile.save(profile_path)
        
        logger.debug(f"Updated profile for {self.current_profile.speaker_id}")
        return self.current_profile
    
    def detect_change(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> bool:
        """Detect if the speaker has changed.
        
        Args:
            audio: Current audio sample
            sample_rate: Sample rate
        
        Returns:
            True if speaker changed, False otherwise
        """
        if self.current_profile is None:
            # No current speaker, consider it a change
            return True
        
        # Extract embedding from current audio
        current_embedding = self.extract_embedding(audio, sample_rate)
        
        # Compare with current profile
        changed = detect_speaker_change(
            self.current_profile.embedding,
            current_embedding,
            self.change_threshold
        )
        
        if changed:
            logger.info("Speaker change detected")
        
        return changed
    
    def identify_speaker(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> Optional[str]:
        """Identify which known speaker is speaking.
        
        Args:
            audio: Audio sample
            sample_rate: Sample rate
        
        Returns:
            Speaker ID of best match, or None if no good match
        """
        if not self.profiles:
            logger.warning("No profiles available for identification")
            return None
        
        # Extract embedding
        current_embedding = self.extract_embedding(audio, sample_rate)
        
        # Find best match
        best_match = None
        best_distance = float('inf')
        
        for speaker_id, profile in self.profiles.items():
            from scipy.spatial.distance import cosine
            distance = cosine(current_embedding, profile.embedding)
            
            if distance < best_distance:
                best_distance = distance
                best_match = speaker_id
        
        # Check if match is good enough
        if best_distance < self.change_threshold:
            logger.info(f"Identified speaker: {best_match} (distance={best_distance:.3f})")
            return best_match
        else:
            logger.info(f"No good match found (best distance={best_distance:.3f})")
            return None
    
    def get_current_profile(self) -> Optional[TimbreProfile]:
        """Get the current active speaker profile.
        
        Returns:
            Current profile or None
        """
        return self.current_profile
    
    def set_current_speaker(self, speaker_id: str) -> bool:
        """Set the current speaker by ID.
        
        Args:
            speaker_id: Speaker identifier
        
        Returns:
            True if successful, False otherwise
        """
        if speaker_id in self.profiles:
            self.current_profile = self.profiles[speaker_id]
            logger.info(f"Set current speaker to: {speaker_id}")
            return True
        else:
            # Try loading from disk
            profile = self.load_profile(speaker_id)
            if profile:
                self.current_profile = profile
                return True
            else:
                logger.warning(f"Speaker not found: {speaker_id}")
                return False
    
    def list_speakers(self) -> list[str]:
        """List all known speaker IDs.
        
        Returns:
            List of speaker IDs
        """
        # Get from memory
        speaker_ids = set(self.profiles.keys())
        
        # Add from disk
        for profile_file in self.profile_dir.glob("*.json"):
            speaker_id = profile_file.stem
            speaker_ids.add(speaker_id)
        
        return sorted(list(speaker_ids))
