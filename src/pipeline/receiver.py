"""Receiver pipeline - Decompress packets and synthesize audio."""

import asyncio
from typing import Optional, Dict
import numpy as np

from src.compression.text import TextCompressor
from src.compression.prosody import ProsodyCompressor
from src.compression.timbre import TimbreCompressor
from src.network import Packet, PacketType
from src.audio import AudioOutput, JitterBuffer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReceiverPipeline:
    """Receive compressed packets and synthesize audio."""
    
    def __init__(
        self,
        config: dict,
        sample_rate: int = 16000
    ):
        """Initialize receiver pipeline.
        
        Args:
            config: Configuration dictionary
            sample_rate: Audio sample rate
        """
        self.config = config
        self.sample_rate = sample_rate
        
        # Initialize components
        # TODO: Use actual TTS implementation when ready
        self.tts = None  # BaseTTS instance will go here
        self.tts_enabled = False
        
        # Decompression
        self.text_compressor = TextCompressor(
            algorithm=config.get("compression", {}).get("text_algorithm", "brotli"),
            level=config.get("compression", {}).get("text_level", 5)
        )
        
        self.prosody_compressor = ProsodyCompressor(
            pitch_bits=config.get("compression", {}).get("prosody_quantization_pitch_bits", 6),
            energy_bits=config.get("compression", {}).get("prosody_quantization_energy_bits", 4),
            rate_bits=config.get("compression", {}).get("prosody_quantization_rate_bits", 4)
        )
        
        self.timbre_compressor = TimbreCompressor()
        
        # Audio output
        self.audio_output = AudioOutput(sample_rate=sample_rate)
        
        # Jitter buffer
        self.jitter_buffer = JitterBuffer(
            sample_rate=sample_rate,
            initial_delay_ms=100,
            max_delay_ms=300
        )
        
        # State
        self.is_running = False
        self.current_speaker_embedding: Optional[np.ndarray] = None
        self.speaker_cache: Dict[int, np.ndarray] = {}
        self.current_prosody: Dict = {}
        
        logger.info("Receiver pipeline initialized")
    
    async def start(self) -> None:
        """Start the receiver pipeline."""
        if self.is_running:
            logger.warning("Pipeline already running")
            return
        
        self.is_running = True
        
        # Start audio output
        self.audio_output.start()
        
        # Start playback task
        asyncio.create_task(self._playback_loop())
        
        logger.info("Receiver pipeline started")
    
    async def receive_packet(self, packet: Packet) -> None:
        """Receive and process a packet.
        
        Args:
            packet: Received packet
        """
        try:
            if packet.packet_type == PacketType.TEXT:
                await self._process_text_packet(packet)
            
            elif packet.packet_type == PacketType.PROSODY:
                await self._process_prosody_packet(packet)
            
            elif packet.packet_type == PacketType.TIMBRE:
                await self._process_speaker_packet(packet)
                
            elif packet.packet_type == PacketType.TIMBRE_PROFILE:
                await self._process_speaker_profile_packet(packet)
            
            elif packet.packet_type == PacketType.CONTROL:
                await self._process_control_packet(packet)
            
        except Exception as e:
            logger.error(f"Error processing packet: {e}")
    
    async def _process_control_packet(self, packet: Packet) -> None:
        """Process control packet.
        
        Args:
            packet: Control packet
        """
        try:
            # TODO: Implement control packet handling (e.g., codec negotiation)
            logger.debug("Received control packet")
        except Exception as e:
            logger.error(f"Error processing control packet: {e}")
    
    async def _process_text_packet(self, packet: Packet) -> None:
        """Process text packet and synthesize speech.
        
        Args:
            packet: Text packet
        """
        try:
            # Decompress text
            text = self.text_compressor.decompress(packet.payload)
            logger.info(f"Received text: {text}")
            
            # TODO: Synthesize speech when TTS is implemented
            if self.tts_enabled and self.tts is not None:
                result = self.tts.synthesize(
                    text=text,
                    speaker_embedding=self.current_speaker_embedding,
                    prosody=self.current_prosody
                )
                
                # Add to jitter buffer for playback
                await self.jitter_buffer.add_packet(
                    sequence=packet.sequence_number,
                    audio_data=result.audio
                )
            else:
                logger.debug("TTS not enabled - text received but not synthesized")
            
        except Exception as e:
            logger.error(f"Error processing text packet: {e}")
    
    async def _process_prosody_packet(self, packet: Packet) -> None:
        """Process prosody packet.
        
        Args:
            packet: Prosody packet
        """
        try:
            # Decompress prosody features
            # TODO: Implement proper prosody decompression and storage
            logger.debug("Received prosody packet")
            
            # Store for next synthesis
            self.current_prosody = {
                "timestamp": packet.timestamp
            }
            
        except Exception as e:
            logger.error(f"Error processing prosody packet: {e}")
    
    async def _process_speaker_packet(self, packet: Packet) -> None:
        """Process speaker embedding packet.
        
        Args:
            packet: Speaker packet
        """
        try:
            # Payload: ID (4 bytes) + Embedding
            import struct
            
            # Check if payload is large enough for ID + minimal embedding
            if len(packet.payload) < 4:
                logger.error("Speaker packet too short")
                return
                
            id_val = struct.unpack('!I', packet.payload[:4])[0]
            embedding_data = packet.payload[4:]
            
            # Decompress speaker embedding (assume 192-dim from ECAPA-TDNN)
            embedding = self.timbre_compressor.decompress(
                embedding_data,
                dim=192
            )
            
            # Cache it
            self.speaker_cache[id_val] = embedding
            self.current_speaker_embedding = embedding
            logger.info(f"Updated speaker embedding (ID={id_val})")
            
        except Exception as e:
            logger.error(f"Error processing speaker packet: {e}")

    async def _process_speaker_profile_packet(self, packet: Packet) -> None:
        """Process speaker profile packet (ID only).
        
        Args:
            packet: Speaker profile packet
        """
        try:
            # Payload: ID (4 bytes)
            import struct
            
            if len(packet.payload) < 4:
                logger.error("Speaker profile packet too short")
                return
                
            id_val = struct.unpack('!I', packet.payload[:4])[0]
            
            if id_val in self.speaker_cache:
                self.current_speaker_embedding = self.speaker_cache[id_val]
                logger.info(f"Switched to cached speaker (ID={id_val})")
            else:
                logger.warning(f"Received profile ID {id_val} but not in cache")
                # Request full profile? (Control packet)
                
        except Exception as e:
            logger.error(f"Error processing speaker profile packet: {e}")
    
    async def _playback_loop(self) -> None:
        """Audio playback loop."""
        while self.is_running:
            try:
                # Get next packet from jitter buffer
                audio_data = await self.jitter_buffer.get_next_packet()
                
                if audio_data is not None:
                    # Play audio
                    self.audio_output.play(audio_data)
                else:
                    # No data available, wait a bit
                    await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in playback loop: {e}")
                await asyncio.sleep(0.1)
    
    async def stop(self) -> None:
        """Stop the receiver pipeline."""
        self.is_running = False
        self.audio_output.stop()
        self.audio_output.close()
        logger.info("Receiver pipeline stopped")
    
    def get_statistics(self) -> dict:
        """Get receiver statistics.
        
        Returns:
            Statistics dictionary
        """
        return self.jitter_buffer.get_statistics()
