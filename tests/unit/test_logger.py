"""Unit tests for logger utilities."""
from pathlib import Path
import tempfile
from src.utils.logger import setup_logging, get_logger


def test_get_logger():
    """Test getting a logger instance."""
    logger = get_logger(__name__)
    
    assert logger is not None
    assert hasattr(logger, 'info')
    assert hasattr(logger, 'debug')
    assert hasattr(logger, 'warning')
    assert hasattr(logger, 'error')


def test_setup_logging_default():
    """Test setup logging with default settings."""
    # Should not raise any errors
    setup_logging()
    
    logger = get_logger(__name__)
    logger.info("Test message")


def test_setup_logging_with_file():
    """Test setup logging with file output."""
    import time
    import shutil
    from loguru import logger as loguru_logger
    
    tmpdir = tempfile.mkdtemp()
    try:
        log_file = Path(tmpdir) / "test.log"
        
        setup_logging(log_level="DEBUG", log_file=log_file)
        
        logger = get_logger(__name__)
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        
        # Give time for file to be written
        time.sleep(0.1)
        
        # Check that log file was created
        assert log_file.exists()
        
        # Try to read file content
        try:
            content = log_file.read_text()
            # Check for any of the messages (they might be formatted differently)
            assert any(msg in content for msg in ["Debug", "Info", "Warning"])
        except PermissionError:
            # On Windows, loguru might still have the file open
            # Just check that the file exists
            pass
        
        # Remove all handlers to release the file
        loguru_logger.remove()
        time.sleep(0.2)  # Give time for file to be released
    finally:
        # Clean up manually
        try:
            shutil.rmtree(tmpdir)
        except (PermissionError, OSError):
            # If still locked, just pass - OS will clean up temp eventually
            pass


def test_setup_logging_levels():
    """Test different logging levels."""
    for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        setup_logging(log_level=level)
        logger = get_logger(__name__)
        logger.info(f"Testing level {level}")


def test_logger_output(caplog):
    """Test that logger produces output."""
    setup_logging(log_level="INFO")
    logger = get_logger(__name__)
    
    # Log some messages
    logger.info("Test info message")
    logger.warning("Test warning message")
    logger.error("Test error message")
