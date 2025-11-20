"""Bandwidth monitoring utilities."""

import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BandwidthStats:
    """Bandwidth statistics."""
    total_bytes: int
    total_packets: int
    current_bps: float
    average_bps: float
    peak_bps: float
    timestamp: float


class BandwidthMonitor:
    """Monitor and track bandwidth usage."""
    
    def __init__(self, window_size: int = 10):
        """Initialize bandwidth monitor.
        
        Args:
            window_size: Number of seconds for moving average
        """
        self.window_size = window_size
        self.samples = deque(maxlen=window_size)
        
        self.total_bytes = 0
        self.total_packets = 0
        self.start_time = time.time()
        self.peak_bps = 0.0
        
        # Per-component tracking
        self.component_bytes: Dict[str, int] = {
            "text": 0,
            "prosody": 0,
            "timbre": 0,
            "overhead": 0,
        }
    
    def record(self, num_bytes: int, component: Optional[str] = None) -> None:
        """Record bandwidth usage.
        
        Args:
            num_bytes: Number of bytes transmitted
            component: Component name (text, prosody, timbre, overhead)
        """
        timestamp = time.time()
        self.samples.append((timestamp, num_bytes))
        
        self.total_bytes += num_bytes
        self.total_packets += 1
        
        if component and component in self.component_bytes:
            self.component_bytes[component] += num_bytes
        
        # Update peak
        current_bps = self.get_current_bps()
        if current_bps > self.peak_bps:
            self.peak_bps = current_bps
    
    def get_current_bps(self) -> float:
        """Get current bandwidth usage in bits per second.
        
        Returns:
            Current bandwidth in bps
        """
        if not self.samples:
            return 0.0
        
        current_time = time.time()
        
        # Calculate bytes in last second
        recent_bytes = sum(
            num_bytes
            for timestamp, num_bytes in self.samples
            if current_time - timestamp <= 1.0
        )
        
        return recent_bytes * 8.0  # Convert to bits
    
    def get_average_bps(self) -> float:
        """Get average bandwidth usage in bits per second.
        
        Returns:
            Average bandwidth in bps
        """
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0.0
        
        return (self.total_bytes * 8.0) / elapsed
    
    def get_stats(self) -> BandwidthStats:
        """Get bandwidth statistics.
        
        Returns:
            Bandwidth statistics
        """
        return BandwidthStats(
            total_bytes=self.total_bytes,
            total_packets=self.total_packets,
            current_bps=self.get_current_bps(),
            average_bps=self.get_average_bps(),
            peak_bps=self.peak_bps,
            timestamp=time.time(),
        )
    
    def get_component_breakdown(self) -> Dict[str, float]:
        """Get bandwidth breakdown by component.
        
        Returns:
            Dictionary of component percentages
        """
        if self.total_bytes == 0:
            return {k: 0.0 for k in self.component_bytes}
        
        return {
            component: (bytes_count / self.total_bytes) * 100.0
            for component, bytes_count in self.component_bytes.items()
        }
    
    def reset(self) -> None:
        """Reset bandwidth statistics."""
        self.samples.clear()
        self.total_bytes = 0
        self.total_packets = 0
        self.start_time = time.time()
        self.peak_bps = 0.0
        self.component_bytes = {k: 0 for k in self.component_bytes}
    
    def log_stats(self) -> None:
        """Log current bandwidth statistics."""
        stats = self.get_stats()
        breakdown = self.get_component_breakdown()
        
        logger.info(
            f"Bandwidth Stats - "
            f"Current: {stats.current_bps:.1f} bps, "
            f"Average: {stats.average_bps:.1f} bps, "
            f"Peak: {stats.peak_bps:.1f} bps"
        )
        logger.debug(
            f"Component Breakdown - "
            f"Text: {breakdown['text']:.1f}%, "
            f"Prosody: {breakdown['prosody']:.1f}%, "
            f"Timbre: {breakdown['timbre']:.1f}%, "
            f"Overhead: {breakdown['overhead']:.1f}%"
        )
