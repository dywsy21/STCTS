"""Unit tests for network packet format."""
import pytest
from src.network.packet import Packet, PacketType

def test_packet_creation():
    """Test creating a packet."""
    packet = Packet(
        packet_type=PacketType.TEXT,
        sequence_number=42,
        timestamp=1000,
        payload=b"test data"
    )
    
    assert packet.packet_type == PacketType.TEXT
    assert packet.sequence_number == 42
    assert packet.timestamp == 1000
    assert packet.payload == b"test data"

def test_packet_serialization():
    """Test packet to_bytes and from_bytes."""
    original = Packet(
        packet_type=PacketType.PROSODY,
        sequence_number=123,
        timestamp=5000,
        payload=b"compressed prosody data"
    )
    
    # Serialize
    data = original.to_bytes()
    
    # Deserialize
    restored = Packet.from_bytes(data)
    
    assert restored.packet_type == original.packet_type
    assert restored.sequence_number == original.sequence_number
    assert restored.timestamp == original.timestamp
    assert restored.payload == original.payload

def test_packet_types():
    """Test all packet types."""
    for ptype in [PacketType.TEXT, PacketType.PROSODY, PacketType.TIMBRE, PacketType.CONTROL]:
        packet = Packet(
            packet_type=ptype,
            sequence_number=1,
            timestamp=100,
            payload=b"test"
        )
        
        data = packet.to_bytes()
        restored = Packet.from_bytes(data)
        
        assert restored.packet_type == ptype

def test_packet_empty_payload():
    """Test packet with empty payload."""
    packet = Packet(
        packet_type=PacketType.CONTROL,
        sequence_number=0,
        timestamp=0,
        payload=b""
    )
    
    data = packet.to_bytes()
    restored = Packet.from_bytes(data)
    
    assert restored.payload == b""

def test_packet_large_payload():
    """Test packet with large payload."""
    large_payload = b"x" * 10000
    packet = Packet(
        packet_type=PacketType.TEXT,
        sequence_number=999,
        timestamp=10000,
        payload=large_payload
    )
    
    data = packet.to_bytes()
    restored = Packet.from_bytes(data)
    
    assert restored.payload == large_payload

def test_packet_max_sequence_number():
    """Test packet with maximum sequence number."""
    packet = Packet(
        packet_type=PacketType.TEXT,
        sequence_number=65535,  # Max 16-bit value
        timestamp=8191,  # Max 13-bit value
        payload=b"test"
    )
    
    data = packet.to_bytes()
    restored = Packet.from_bytes(data)
    
    assert restored.sequence_number == 65535
    assert restored.timestamp == 8191

def test_packet_header_size():
    """Test that packet header is exactly 4 bytes."""
    packet = Packet(
        packet_type=PacketType.TEXT,
        sequence_number=1,
        timestamp=1,
        payload=b"test"
    )
    
    data = packet.to_bytes()
    # Header (5 bytes) + payload length
    expected_size = Packet.HEADER_SIZE + len(packet.payload)
    assert len(data) == expected_size
