"""Opus codec baseline."""

import numpy as np
import subprocess
import tempfile
import os
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def check_opus_available() -> tuple:
    """Check if Opus tools are available.
    
    Returns:
        Tuple of (available: bool, method: str, message: str)
    """
    logger.info("Checking Opus availability...")
    
    # Try Python opuslib first
    try:
        import opuslib
        logger.info("opuslib imported successfully, testing encoder creation...")
        # Actually test if it can create an encoder
        try:
            _ = opuslib.Encoder(16000, 1, opuslib.APPLICATION_AUDIO)
            logger.info("✓ opuslib encoder created successfully")
            return True, "opuslib", "Using Python opuslib"
        except Exception as e:
            logger.warning(f"✗ opuslib imported but failed to create encoder: {e}")
            logger.warning("This usually means Opus native library is not installed")
            logger.info("Falling back to system opus-tools...")
    except ImportError as e:
        logger.info(f"opuslib not installed: {e}, trying opus-tools...")
    except Exception as e:
        # Catch opuslib's "Could not find Opus library" exception during import
        logger.warning(f"✗ opuslib failed to load: {e}")
        logger.info("Falling back to system opus-tools...")
    
    # Try system opus-tools
    try:
        logger.info("Testing opusenc command...")
        result = subprocess.run(
            ["opusenc", "--version"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            logger.info(f"✓ opus-tools found: {result.stdout.decode().strip()}")
            return True, "opus-tools", "Using system opus-tools"
        else:
            logger.warning(f"✗ opusenc returned non-zero: {result.returncode}")
    except FileNotFoundError:
        logger.warning("✗ opusenc command not found in PATH")
    except subprocess.SubprocessError as e:
        logger.warning(f"✗ Error running opusenc: {e}")
    
    logger.error("✗ Opus not available (neither opuslib nor opus-tools)")
    return False, None, "Opus not available"


class OpusCodec:
    """Opus codec for baseline comparison."""
    
    def __init__(self, bitrate: int = 6000, sample_rate: int = 16000):
        """Initialize Opus codec.
        
        Args:
            bitrate: Target bitrate in bps
            sample_rate: Audio sample rate (8000, 12000, 16000, 24000, or 48000)
        """
        self.bitrate = bitrate
        self.sample_rate = sample_rate
        
        # Check which method is available
        self.available, self.method, msg = check_opus_available()
        if self.available:
            logger.info(f"Opus baseline: {msg}")
        else:
            import platform
            logger.warning("⚠️  Opus codec not available!")
            
            if platform.system() == "Windows":
                logger.warning("")
                logger.warning("   📦 Windows Installation Options:")
                logger.warning("   Option 1 (Easiest): Use system opus-tools")
                logger.warning("     1. You already have opusenc/opusdec in PATH")
                logger.warning("     2. Benchmark will use those instead")
                logger.warning("     3. This should work now!")
                logger.warning("")
                logger.warning("   Option 2: Install opuslib + Opus DLL")
                logger.warning("     1. pip install opuslib")
                logger.warning("     2. Download Opus DLL from https://opus-codec.org/downloads/")
                logger.warning("     3. Place opus.dll in system PATH")
            else:
                logger.warning("   Linux: sudo apt install opus-tools libopus0")
                logger.warning("   macOS: brew install opus-tools")
    
    def encode_decode_opuslib(self, audio: np.ndarray) -> tuple:
        """Encode and decode using Python opuslib.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Tuple of (reconstructed_audio, compressed_size_bytes)
        """
        try:
            import opuslib
            
            # Convert to int16
            audio_int16 = (audio * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            # Create encoder
            encoder = opuslib.Encoder(
                self.sample_rate,
                1,  # channels
                opuslib.APPLICATION_AUDIO
            )
            # Opus minimum bitrate is 6000 bps (6 kbps)
            encoder.bitrate = max(6000, self.bitrate)
            
            # Encode in chunks (Opus requires specific frame sizes)
            frame_size = int(self.sample_rate * 0.020)  # 20ms frames
            encoded_packets = []
            
            for i in range(0, len(audio_int16), frame_size):
                frame = audio_int16[i:i + frame_size]
                
                # Pad last frame if needed
                if len(frame) < frame_size:
                    frame = np.pad(frame, (0, frame_size - len(frame)))
                
                frame_bytes = frame.tobytes()
                encoded = encoder.encode(frame_bytes, frame_size)
                encoded_packets.append(encoded)
            
            # Calculate compressed size
            compressed_size = sum(len(p) for p in encoded_packets)
            
            # Decode
            decoder = opuslib.Decoder(self.sample_rate, 1)
            decoded_frames = []
            
            for packet in encoded_packets:
                decoded = decoder.decode(packet, frame_size)
                decoded_int16 = np.frombuffer(decoded, dtype=np.int16)
                decoded_frames.append(decoded_int16)
            
            # Concatenate and convert back to float
            reconstructed = np.concatenate(decoded_frames)
            reconstructed = reconstructed[:len(audio_int16)]  # Trim padding
            reconstructed = reconstructed.astype(np.float32) / 32767.0
            
            logger.info(f"Opus: {len(audio)/self.sample_rate:.1f}s audio -> {compressed_size}B")
            
            return reconstructed, compressed_size
            
        except Exception as e:
            logger.error(f"Error in opuslib: {e}")
            return audio, 0
    
    def encode_decode_tools(self, audio: np.ndarray) -> tuple:
        """Encode and decode using system opus-tools.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Tuple of (reconstructed_audio, compressed_size_bytes)
        """
        import soundfile as sf
        
        # Create temp files
        with tempfile.TemporaryDirectory() as tmpdir:
            input_wav = Path(tmpdir) / "input.wav"
            opus_file = Path(tmpdir) / "compressed.opus"
            output_wav = Path(tmpdir) / "output.wav"
            
            # Save input audio
            sf.write(input_wav, audio, self.sample_rate)
            
            try:
                # Encode to Opus
                # Convert bps to kbps, with minimum of 6 kbps (Opus minimum)
                bitrate_kbps = max(6, self.bitrate // 1000)
                encode_cmd = [
                    "opusenc",
                    "--bitrate", str(bitrate_kbps),
                    "--quiet",
                    str(input_wav),
                    str(opus_file)
                ]
                subprocess.run(encode_cmd, check=True, capture_output=True)
                
                # Get compressed size
                compressed_size = os.path.getsize(opus_file)
                
                # Decode from Opus
                decode_cmd = [
                    "opusdec",
                    "--quiet",
                    str(opus_file),
                    str(output_wav)
                ]
                subprocess.run(decode_cmd, check=True, capture_output=True)
                
                # Load reconstructed audio
                reconstructed, sr = sf.read(output_wav)
                
                # Resample if needed
                if sr != self.sample_rate:
                    import librosa
                    reconstructed = librosa.resample(
                        reconstructed,
                        orig_sr=sr,
                        target_sr=self.sample_rate
                    )
                
                logger.info(f"Opus: {len(audio)/self.sample_rate:.1f}s audio -> {compressed_size}B")
                
                return reconstructed, compressed_size
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Opus encoding/decoding failed: {e}")
                logger.error("Make sure opus-tools is installed: sudo apt install opus-tools")
                return audio, 0
            except Exception as e:
                logger.error(f"Error in Opus codec: {e}")
                return audio, 0
    
    def encode_decode(self, audio: np.ndarray) -> tuple:
        """Encode and decode audio using Opus.
        
        Automatically uses the best available method:
        1. opuslib (Python library) - preferred
        2. opus-tools (system command) - fallback
        
        Args:
            audio: Input audio signal
            
        Returns:
            Tuple of (reconstructed_audio, compressed_size_bytes)
        """
        if not self.available:
            logger.error("❌ Opus codec not available - returning original audio")
            logger.error("   Install opuslib: pip install opuslib")
            return audio, 0
        
        logger.info(f"🎵 Using Opus method: {self.method}")
        
        if self.method == "opuslib":
            return self.encode_decode_opuslib(audio)
        elif self.method == "opus-tools":
            return self.encode_decode_tools(audio)
        else:
            logger.error(f"❌ Unknown Opus method: {self.method}")
            return audio, 0
