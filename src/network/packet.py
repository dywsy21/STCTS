"""Packet format definition."""

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PacketType(IntEnum):
    """Packet types."""
    TEXT = 0
    PROSODY = 1
    TIMBRE = 2
    CONTROL = 3
    TIMBRE_PROFILE = 4


@dataclass
class Packet:
    """Network packet."""
    packet_type: PacketType
    sequence_number: int
    timestamp: int  # Relative timestamp (wraps every 8 seconds)
    payload: bytes
    
    HEADER_SIZE = 5  # bytes (1 byte + 2 bytes + 2 bytes for !BHH format)
    MAX_PAYLOAD_SIZE = 1400  # MTU consideration
    
    def to_bytes(self) -> bytes:
        """Serialize packet to bytes.
        
        Header format (4 bytes):
        - Version (2 bits): 0
        - Packet Type (3 bits): 0-7
        - Reserved (3 bits): 0
        - Sequence Number (16 bits): 0-65535
        - Timestamp (13 bits, relative, in lower 2 bytes): wraps every ~8s @ 1kHz
        
        Returns:
            Serialized packet
        """
        # Pack first byte: version (2 bits) + type (3 bits) + reserved (3 bits)
        version = 0
        first_byte = (version << 6) | (self.packet_type << 3)
        
        # Pack header: first_byte, seq_num (16 bits), timestamp (13 bits in 2 bytes)
        timestamp_masked = self.timestamp & 0x1FFF  # 13 bits
        
        header = struct.pack(
            "!BHH",
            first_byte,
            self.sequence_number,
            timestamp_masked,
        )
        
        return header + self.payload
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "Packet":
        """Deserialize packet from bytes.
        
        Args:
            data: Serialized packet
        
        Returns:
            Packet object
        """
        if len(data) < cls.HEADER_SIZE:
            raise ValueError("Data too short for packet header")
        
        # Unpack header
        first_byte, seq_num, timestamp_masked = struct.unpack("!BHH", data[:cls.HEADER_SIZE])
        
        # Extract fields
        packet_type = PacketType((first_byte >> 3) & 0x07)
        timestamp = timestamp_masked & 0x1FFF
        
        # Extract payload
        payload = data[cls.HEADER_SIZE:]
        
        return cls(
            packet_type=packet_type,
            sequence_number=seq_num,
            timestamp=timestamp,
            payload=payload,
        )
    
    def __len__(self) -> int:
        """Get total packet size."""
        return self.HEADER_SIZE + len(self.payload)
