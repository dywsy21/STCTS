"""Sender pipeline - Audio to compressed packets."""

import asyncio
from typing import Optional
import numpy as np
import time

from src.stt import FasterWhisperSTT
from src.prosody import ProsodyExtractor, ProsodyFeatures
from src.speaker import SpeakerManager
from src.compression.text import TextCompressor
from src.compression.prosody import ProsodyCompressor
from src.compression.timbre import TimbreCompressor
from src.network import Packet, PacketType, PacketPriorityQueue
from src.audio import AudioInput
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SenderPipeline:
    """Process audio input and send compressed packets.
    
    This pipeline supports real-time streaming by:
    1. Buffering audio chunks from microphone
    2. Processing in background when enough data accumulated
    3. Sending packets asynchronously without blocking
    """
    
    def __init__(
        self,
        config: dict,
        packet_queue: PacketPriorityQueue,
        sample_rate: int = 16000,
        is_transmitting_callback=None
    ):
        """Initialize sender pipeline.
        
        Args:
            config: Configuration dictionary
            packet_queue: Packet queue for transmission
            sample_rate: Audio sample rate
            is_transmitting_callback: Callable that returns bool indicating if transmission is active
        """
        self.config = config
        self.packet_queue = packet_queue
        self.sample_rate = sample_rate
        self.is_transmitting_callback = is_transmitting_callback
        
        # Initialize components
        device = config.get("device", "auto")
        
        self.stt = FasterWhisperSTT(
            model_size=config.get("stt", {}).get("model_size", "small"),
            device=device
        )
        
        self.prosody_extractor = ProsodyExtractor(
            sample_rate=sample_rate,
            enable_emotion=True,
            enable_emphasis=True
        )
        
        self.speaker_manager = SpeakerManager(
            change_threshold=config.get("speaker", {}).get("change_threshold", 0.3),
            device=device
        )
        
        # Compression
        self.text_compressor = TextCompressor(
            algorithm=config.get("compression", {}).get("text_algorithm", "brotli"),
            level=config.get("compression", {}).get("text_level", 5)
        )
        
        self.prosody_compressor = ProsodyCompressor(
            pitch_bits=config.get("compression", {}).get("prosody_quantization_pitch_bits", 6),
            energy_bits=config.get("compression", {}).get("prosody_quantization_energy_bits", 4),
            rate_bits=config.get("compression", {}).get("prosody_quantization_rate_bits", 4),
            keyframe_rate=config.get("prosody", {}).get("update_rate_hz", 1.0)
        )
        
        self.timbre_compressor = TimbreCompressor()
        
        # Audio input
        self.audio_input = AudioInput(
            sample_rate=sample_rate,
            chunk_size=config.get("audio", {}).get("chunk_size", 1024)
        )
        
        # State
        self.audio_buffer = []
        self.is_running = False
        self.sequence_number = 0
        self.timbre_sent = False  # Track if initial timbre has been sent
        
        logger.info("Sender pipeline initialized")
    
    async def start(self) -> None:
        """Start the sender pipeline."""
        if self.is_running:
            logger.warning("Pipeline already running")
            return
        
        self.is_running = True
        
        # Start audio input
        self.audio_input.start(self._on_audio_chunk)
        
        # Start processing task
        asyncio.create_task(self._processing_loop())
        
        logger.info("Sender pipeline started")
    
    def _on_audio_chunk(self, audio_data: np.ndarray) -> None:
        """Handle incoming audio chunk.
        
        Args:
            audio_data: Audio samples
        """
        self.audio_buffer.append(audio_data)
    
    async def _processing_loop(self) -> None:
        """Main processing loop."""
        while self.is_running:
            try:
                # Check if transmission is active (push-to-talk)
                if self.is_transmitting_callback and not self.is_transmitting_callback():
                    # Not transmitting - clear buffer and wait
                    self.audio_buffer.clear()
                    await asyncio.sleep(0.1)
                    continue
                
                # Wait for enough audio data (require at least 1 second of audio)
                if len(self.audio_buffer) < 50:  # ~1 second at 1024 chunk size, 16kHz
                    await asyncio.sleep(0.01)
                    continue
                
                # Get audio data
                audio_data = np.concatenate(self.audio_buffer)
                self.audio_buffer.clear()
                
                # Process the audio
                await self._process_audio(audio_data)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_audio(self, audio_data: np.ndarray) -> None:
        """Process audio and create packets.
        
        Args:
            audio_data: Audio samples
        """
        # 1. STT - Transcribe audio
        result = self.stt.transcribe(audio_data, self.sample_rate)
        if not result or not result.text.strip():
            return  # No speech detected
        
        text = result.text.strip()
        logger.info(f"Transcribed: {text}")
        
        # 2. Extract prosody features
        prosody_features = self.prosody_extractor.extract_all(audio_data)
        
        # 3. Check for speaker change
        speaker_changed = self.speaker_manager.detect_change(audio_data, self.sample_rate)
        
        # 4. Create and send text packet
        await self._send_text_packet(text)
        
        # 5. Create and send prosody packet
        await self._send_prosody_packet(prosody_features)
        
        # 6. Send speaker embedding on first transmission or when speaker changes
        if not self.timbre_sent or speaker_changed:
            await self._send_speaker_packet(audio_data)
            self.timbre_sent = True
    
    async def _send_text_packet(self, text: str) -> None:
        """Compress and send text packet.
        
        Args:
            text: Text to send
        """
        try:
            # Compress text
            compressed_text = self.text_compressor.compress(text)
            
            # Create packet
            packet = Packet(
                packet_type=PacketType.TEXT,
                sequence_number=self.sequence_number,
                timestamp=int(time.time() * 1000) & 0xFFFF,  # Convert to ms, mask to 16 bits
                payload=compressed_text
            )
            
            self.sequence_number += 1
            
            # Add to queue with high priority
            await self.packet_queue.put(packet, PacketPriorityQueue.PRIORITY_HIGH)
            
            logger.debug(f"Sent text packet ({len(compressed_text)} bytes)")
            
        except Exception as e:
            logger.error(f"Error sending text packet: {e}")
    
    async def _send_prosody_packet(self, features: ProsodyFeatures) -> None:
        """Compress and send prosody packet.
        
        Args:
            features: Prosody features
        """
        try:
            # Compress prosody features
            compressed_prosody = b""
            
            if features.pitch is not None:
                compressed_prosody += self.prosody_compressor.compress_pitch(features.pitch)
            
            if features.energy is not None:
                compressed_prosody += self.prosody_compressor.compress_energy(features.energy)
            
            # Create packet
            packet = Packet(
                packet_type=PacketType.PROSODY,
                sequence_number=self.sequence_number,
                timestamp=int(time.time() * 1000) & 0xFFFF,  # Convert to ms, mask to 16 bits
                payload=compressed_prosody
            )
            
            self.sequence_number += 1
            
            # Add to queue with medium priority
            await self.packet_queue.put(packet, PacketPriorityQueue.PRIORITY_MEDIUM)
            
            logger.debug(f"Sent prosody packet ({len(compressed_prosody)} bytes)")
            
        except Exception as e:
            logger.error(f"Error sending prosody packet: {e}")
    
    async def _send_speaker_packet(self, audio_data: np.ndarray) -> None:
        """Extract and send speaker embedding packet.
        
        Args:
            audio_data: Audio samples
        """
        try:
            # Extract speaker embedding
            embedding = self.speaker_manager.extract_embedding(audio_data, self.sample_rate)
            
            # Identify speaker
            speaker_id = self.speaker_manager.identify_from_embedding(embedding)
            
            import struct
            import zlib
            
            if speaker_id:
                # Known speaker - send ID only
                try:
                    id_val = int(speaker_id)
                except ValueError:
                    # Use CRC32 for non-numeric IDs to get 4 bytes
                    id_val = zlib.crc32(speaker_id.encode())
                
                payload = struct.pack('!I', id_val)
                packet_type = PacketType.TIMBRE_PROFILE
                logger.debug(f"Known speaker {speaker_id}, sending profile ID")
                
            else:
                # New speaker - create profile and send full embedding
                # Generate new ID
                # Use a simple counter based on existing profiles + timestamp to avoid collisions
                # or just find the next available integer ID
                existing_ids = []
                for pid in self.speaker_manager.profiles:
                    try:
                        existing_ids.append(int(pid))
                    except ValueError:
                        pass
                
                new_id = max(existing_ids) + 1 if existing_ids else 1
                speaker_id = str(new_id)
                
                # Create profile
                self.speaker_manager.create_profile(
                    speaker_id=speaker_id,
                    embedding=embedding,
                    sample_rate=self.sample_rate
                )
                
                # Compress embedding
                compressed_embedding = self.timbre_compressor.compress(embedding)
                
                # Payload: ID (4 bytes) + Embedding
                id_val = int(speaker_id)
                payload = struct.pack('!I', id_val) + compressed_embedding
                packet_type = PacketType.TIMBRE
                logger.debug(f"New speaker {speaker_id}, sending full embedding")

            # Create packet
            packet = Packet(
                packet_type=packet_type,
                sequence_number=self.sequence_number,
                timestamp=int(time.time() * 1000) & 0xFFFF,  # Convert to ms, mask to 16 bits
                payload=payload
            )
            
            self.sequence_number += 1
            
            # Add to queue with low priority (sent infrequently)
            await self.packet_queue.put(packet, PacketPriorityQueue.PRIORITY_LOW)
            
            logger.debug(f"Sent speaker packet ({len(payload)} bytes)")
            
        except Exception as e:
            logger.error(f"Error sending speaker packet: {e}")
    
    async def stop(self) -> None:
        """Stop the sender pipeline."""
        self.is_running = False
        self.audio_input.stop()
        self.audio_input.close()
        logger.info("Sender pipeline stopped")
