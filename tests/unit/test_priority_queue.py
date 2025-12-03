"""Unit tests for priority queue."""
import pytest
import asyncio
from src.network.priority_queue import PacketPriorityQueue, PriorityPacket
from src.network.packet import Packet, PacketType


@pytest.mark.asyncio
async def test_priority_queue_creation():
    """Test creating a priority queue."""
    queue = PacketPriorityQueue(max_size=100)
    
    assert queue.max_size == 100
    assert len(queue._queue) == 0


@pytest.mark.asyncio
async def test_priority_queue_put_get():
    """Test adding and retrieving packets."""
    queue = PacketPriorityQueue(max_size=10)
    
    packet = Packet(
        packet_type=PacketType.TEXT,
        sequence_number=1,
        timestamp=100,
        payload=b"test"
    )
    
    # Add packet
    result = await queue.put(packet, priority=PacketPriorityQueue.PRIORITY_HIGH)
    assert result is True
    
    # Get packet
    retrieved = await queue.get(timeout=1.0)
    assert retrieved is not None
    assert retrieved.packet_type == PacketType.TEXT
    assert retrieved.payload == b"test"


@pytest.mark.asyncio
async def test_priority_queue_ordering():
    """Test that packets are retrieved by priority."""
    queue = PacketPriorityQueue(max_size=10)
    
    # Add packets with different priorities
    low_packet = Packet(PacketType.TIMBRE, 1, 100, b"low")
    med_packet = Packet(PacketType.PROSODY, 2, 100, b"medium")
    high_packet = Packet(PacketType.TEXT, 3, 100, b"high")
    
    await queue.put(low_packet, PacketPriorityQueue.PRIORITY_LOW)
    await queue.put(med_packet, PacketPriorityQueue.PRIORITY_MEDIUM)
    await queue.put(high_packet, PacketPriorityQueue.PRIORITY_HIGH)
    
    # Should get high priority first
    p1 = await queue.get(timeout=1.0)
    assert p1.payload == b"high"
    
    # Then medium
    p2 = await queue.get(timeout=1.0)
    assert p2.payload == b"medium"
    
    # Then low
    p3 = await queue.get(timeout=1.0)
    assert p3.payload == b"low"


@pytest.mark.asyncio
async def test_priority_queue_full():
    """Test queue full behavior."""
    queue = PacketPriorityQueue(max_size=2)
    
    packet1 = Packet(PacketType.TEXT, 1, 100, b"test1")
    packet2 = Packet(PacketType.TEXT, 2, 100, b"test2")
    packet3 = Packet(PacketType.TEXT, 3, 100, b"test3")
    
    # Add two packets
    assert await queue.put(packet1) is True
    assert await queue.put(packet2) is True
    
    # Third should fail
    assert await queue.put(packet3) is False


@pytest.mark.asyncio
async def test_priority_queue_timeout():
    """Test get with timeout on empty queue."""
    queue = PacketPriorityQueue(max_size=10)
    
    # Should timeout
    packet = await queue.get(timeout=0.1)
    assert packet is None


@pytest.mark.asyncio
async def test_priority_queue_size():
    """Test getting queue size."""
    queue = PacketPriorityQueue(max_size=10)
    
    assert queue.size() == 0
    
    packet = Packet(PacketType.TEXT, 1, 100, b"test")
    await queue.put(packet)
    
    assert queue.size() == 1


@pytest.mark.asyncio
async def test_priority_queue_empty():
    """Test checking if queue is empty."""
    queue = PacketPriorityQueue(max_size=10)
    
    assert queue.size() == 0
    
    packet = Packet(PacketType.TEXT, 1, 100, b"test")
    await queue.put(packet)
    
    assert queue.size() > 0


@pytest.mark.asyncio
async def test_priority_queue_clear():
    """Test clearing the queue."""
    queue = PacketPriorityQueue(max_size=10)
    
    # Add some packets
    for i in range(5):
        packet = Packet(PacketType.TEXT, i, 100, b"test")
        await queue.put(packet)
    
    assert queue.size() == 5
    
    await queue.clear()
    
    assert queue.size() == 0


@pytest.mark.asyncio
async def test_priority_packet_ordering():
    """Test PriorityPacket comparison."""
    packet1 = Packet(PacketType.TEXT, 1, 100, b"test1")
    packet2 = Packet(PacketType.TEXT, 2, 100, b"test2")
    
    pp1 = PriorityPacket(priority=1, timestamp=1.0, packet=packet1)
    pp2 = PriorityPacket(priority=0, timestamp=2.0, packet=packet2)
    
    # Lower priority value should be "less than"
    assert pp2 < pp1
