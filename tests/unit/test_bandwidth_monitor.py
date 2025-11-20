"""Unit tests for bandwidth monitor."""
import pytest
import time
from src.utils.bandwidth_monitor import BandwidthMonitor

def test_bandwidth_monitor_creation():
    """Test creating a bandwidth monitor."""
    monitor = BandwidthMonitor(window_size=5)
    assert monitor.get_current_bps() == 0

def test_bandwidth_monitor_record():
    """Test recording data."""
    monitor = BandwidthMonitor()
    
    monitor.record(100, component='text')
    monitor.record(200, component='prosody')
    
    # Should have recorded 300 bytes total
    breakdown = monitor.get_component_breakdown()
    assert breakdown['text'] > 0
    assert breakdown['prosody'] > 0

def test_bandwidth_monitor_bps_calculation():
    """Test bits per second calculation."""
    monitor = BandwidthMonitor(window_size=1)
    
    # Record 100 bytes
    monitor.record(100)
    
    # BPS should be approximately 800 bps (100 bytes * 8 bits)
    bps = monitor.get_current_bps()
    assert 700 <= bps <= 900  # Allow some variance

def test_bandwidth_monitor_average():
    """Test average bandwidth calculation."""
    monitor = BandwidthMonitor()
    
    monitor.record(50)
    time.sleep(0.1)
    monitor.record(50)
    time.sleep(0.1)
    monitor.record(50)
    
    avg = monitor.get_average_bps()
    assert avg > 0

def test_bandwidth_monitor_component_breakdown():
    """Test component breakdown."""
    monitor = BandwidthMonitor()
    
    monitor.record(100, component='text')
    monitor.record(200, component='prosody')
    monitor.record(50, component='timbre')
    monitor.record(30, component='overhead')
    
    breakdown = monitor.get_component_breakdown()
    
    assert 'text' in breakdown
    assert 'prosody' in breakdown
    assert 'timbre' in breakdown
    assert 'overhead' in breakdown

def test_bandwidth_monitor_window():
    """Test that old data is removed from window."""
    monitor = BandwidthMonitor(window_size=1)
    
    monitor.record(100)
    time.sleep(1.1)  # Wait longer than 1 second (get_current_bps looks at last 1 second)
    
    # Should have dropped old data (older than 1 second)
    bps = monitor.get_current_bps()
    assert bps == 0  # Data should be completely aged out

def test_bandwidth_monitor_stats():
    """Test getting full statistics."""
    monitor = BandwidthMonitor()
    
    for i in range(10):
        monitor.record(50)
        time.sleep(0.05)
    
    stats = monitor.get_stats()
    
    # stats is a BandwidthStats dataclass, not a dict
    assert hasattr(stats, 'current_bps')
    assert hasattr(stats, 'average_bps')
    assert hasattr(stats, 'total_bytes')
    assert stats.total_bytes == 500
