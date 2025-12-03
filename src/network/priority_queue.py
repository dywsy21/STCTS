"""Priority queue for packet transmission."""

import asyncio
import heapq
from typing import Optional, List
from dataclasses import dataclass, field
import time

from src.network.packet import Packet
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(order=True)
class PriorityPacket:
    """Packet with priority for queue."""
    priority: int = field(compare=True)
    timestamp: float = field(compare=True)
    packet: Packet = field(compare=False)


class PacketPriorityQueue:
    """Priority queue for managing packet transmission."""
    
    # Priority levels
    PRIORITY_HIGH = 0      # Text, critical metadata
    PRIORITY_MEDIUM = 1    # Prosody features
    PRIORITY_LOW = 2       # Speaker embeddings (sent once)
    
    def __init__(self, max_size: int = 1000):
        """Initialize priority queue.
        
        Args:
            max_size: Maximum queue size
        """
        self.max_size = max_size
        self._queue: List[PriorityPacket] = []
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Event()
        
        logger.info(f"Packet priority queue initialized (max_size={max_size})")
    
    async def put(
        self,
        packet: Packet,
        priority: int = PRIORITY_MEDIUM
    ) -> bool:
        """Add packet to queue.
        
        Args:
            packet: Packet to add
            priority: Priority level (0=high, 1=medium, 2=low)
        
        Returns:
            True if added, False if queue full
        """
        async with self._lock:
            if len(self._queue) >= self.max_size:
                logger.warning("Queue full, dropping packet")
                return False
            
            priority_packet = PriorityPacket(
                priority=priority,
                timestamp=time.time(),
                packet=packet
            )
            
            heapq.heappush(self._queue, priority_packet)
            self._not_empty.set()
            
            logger.debug(f"Added packet (priority={priority}, queue_size={len(self._queue)})")
            return True
    
    async def get(self, timeout: Optional[float] = None) -> Optional[Packet]:
        """Get highest priority packet from queue.
        
        Args:
            timeout: Timeout in seconds
        
        Returns:
            Packet or None if timeout
        """
        try:
            # Wait for packet with timeout
            await asyncio.wait_for(
                self._not_empty.wait(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None
        
        async with self._lock:
            if not self._queue:
                self._not_empty.clear()
                return None
            
            priority_packet = heapq.heappop(self._queue)
            
            if not self._queue:
                self._not_empty.clear()
            
            logger.debug(f"Retrieved packet (priority={priority_packet.priority}, queue_size={len(self._queue)})")
            return priority_packet.packet
    
    async def get_batch(
        self,
        max_count: int = 10,
        timeout: Optional[float] = None
    ) -> List[Packet]:
        """Get multiple packets at once.
        
        Args:
            max_count: Maximum number of packets to get
            timeout: Timeout in seconds
        
        Returns:
            List of packets
        """
        packets = []
        
        # Get first packet with timeout
        first_packet = await self.get(timeout=timeout)
        if first_packet is None:
            return packets
        
        packets.append(first_packet)
        
        # Get remaining packets without blocking
        for _ in range(max_count - 1):
            packet = await self.get(timeout=0.001)  # Very short timeout
            if packet is None:
                break
            packets.append(packet)
        
        logger.debug(f"Retrieved batch of {len(packets)} packets")
        return packets
    
    def size(self) -> int:
        """Get current queue size.
        
        Returns:
            Number of packets in queue
        """
        return len(self._queue)
    
    def is_empty(self) -> bool:
        """Check if queue is empty.
        
        Returns:
            True if empty
        """
        return len(self._queue) == 0
    
    def is_full(self) -> bool:
        """Check if queue is full.
        
        Returns:
            True if full
        """
        return len(self._queue) >= self.max_size
    
    async def clear(self) -> None:
        """Clear all packets from queue."""
        async with self._lock:
            self._queue.clear()
            self._not_empty.clear()
            logger.info("Queue cleared")


class CongestionController:
    """Control transmission rate based on network conditions."""
    
    def __init__(
        self,
        initial_rate_bps: int = 450,
        min_rate_bps: int = 300,
        max_rate_bps: int = 600,
        adaptation_interval: float = 1.0
    ):
        """Initialize congestion controller.
        
        Args:
            initial_rate_bps: Initial bitrate
            min_rate_bps: Minimum bitrate
            max_rate_bps: Maximum bitrate
            adaptation_interval: How often to adapt rate (seconds)
        """
        self.current_rate_bps = initial_rate_bps
        self.min_rate_bps = min_rate_bps
        self.max_rate_bps = max_rate_bps
        self.adaptation_interval = adaptation_interval
        
        self.last_adaptation = time.time()
        self.packet_loss_rate = 0.0
        self.rtt_ms = 100.0  # Round-trip time
        
        logger.info(f"Congestion controller initialized (rate={initial_rate_bps} bps)")
    
    def report_packet_loss(self, loss_rate: float) -> None:
        """Report packet loss rate.
        
        Args:
            loss_rate: Loss rate (0.0 to 1.0)
        """
        self.packet_loss_rate = loss_rate
        logger.debug(f"Packet loss rate: {loss_rate:.2%}")
    
    def report_rtt(self, rtt_ms: float) -> None:
        """Report round-trip time.
        
        Args:
            rtt_ms: RTT in milliseconds
        """
        self.rtt_ms = rtt_ms
        logger.debug(f"RTT: {rtt_ms:.1f} ms")
    
    def adapt_rate(self) -> int:
        """Adapt transmission rate based on conditions.
        
        Returns:
            New bitrate in bps
        """
        now = time.time()
        if now - self.last_adaptation < self.adaptation_interval:
            return self.current_rate_bps
        
        self.last_adaptation = now
        
        # Simple adaptation algorithm
        if self.packet_loss_rate > 0.05:  # >5% loss
            # Reduce rate
            self.current_rate_bps = max(
                self.min_rate_bps,
                int(self.current_rate_bps * 0.8)
            )
            logger.info(f"Reducing bitrate to {self.current_rate_bps} bps (high loss)")
        
        elif self.packet_loss_rate < 0.01 and self.rtt_ms < 200:  # <1% loss, good RTT
            # Increase rate
            self.current_rate_bps = min(
                self.max_rate_bps,
                int(self.current_rate_bps * 1.1)
            )
            logger.info(f"Increasing bitrate to {self.current_rate_bps} bps (good conditions)")
        
        return self.current_rate_bps
    
    def get_current_rate(self) -> int:
        """Get current transmission rate.
        
        Returns:
            Bitrate in bps
        """
        return self.current_rate_bps
    
    def should_drop_packet(self, priority: int) -> bool:
        """Determine if packet should be dropped due to congestion.
        
        Args:
            priority: Packet priority
        
        Returns:
            True if should drop
        """
        # Never drop high priority packets
        if priority == PacketPriorityQueue.PRIORITY_HIGH:
            return False
        
        # Drop low priority packets under high congestion
        if priority == PacketPriorityQueue.PRIORITY_LOW and self.packet_loss_rate > 0.1:
            return True
        
        # Drop medium priority under extreme congestion
        if priority == PacketPriorityQueue.PRIORITY_MEDIUM and self.packet_loss_rate > 0.15:
            return True
        
        return False
