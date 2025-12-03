"""
Comprehensive benchmark runner for STT-Compress-TTS.

Usage:
    # Basic usage (uses audio file as reference)
    uv run python -m benchmarks.run_all --config minimal_mode --audio test_audios/tpo53-1.wav
    
    # With custom reference audio for voice cloning
    uv run python -m benchmarks.run_all -c minimal_mode -a test_audios/tpo53-1.wav -r test_audios/reference.wav
    
    # With specific baselines and output file
    uv run python -m benchmarks.run_all -c balanced_mode -a audio.wav -b opus encodec --output results.json
    
    # Run parameter sweep with plot configs (plot_*_mode.yaml)
    uv run python -m benchmarks.run_all --plot --audio test_audios/tpo53-1.wav --output plot_results.json
    
    # Noise resilience testing (inject bit errors into compressed bitstream)
    uv run python -m benchmarks.run_all --audio test_audios/tpo53-1.wav --noise high_quality
    # Tests bit error rates: 0.1% (1e-3), 1% (1e-2), 10% (1e-1)
    
    # LibriSpeech batch testing (optimized with model caching)
    uv run python -m benchmarks.run_all --librispeech 10 --output librispeech_results.json
    uv run python -m benchmarks.run_all --librispeech 20 --noise high_quality --output librispeech_noise.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import time

from src.utils.logger import setup_logging, get_logger
from src.utils.config import load_config

logger = get_logger(__name__)


class ModelManager:
    """Manages model instances for batch processing to avoid repeated loading."""
    
    def __init__(self, config: Any):
        """Initialize model manager with given config."""
        self.config = config
        self._stt = None
        self._tts = None
        self._prosody_extractor = None
        self._speaker_model = None
        self._text_compressor = None
        self._prosody_compressor = None
        self._timbre_compressor = None
        
    def get_stt(self):
        """Get or create STT model."""
        if self._stt is None:
            from src.stt import FasterWhisperSTT
            logger.info("Loading STT model (once)...")
            self._stt = FasterWhisperSTT(
                model_size=self.config.stt.model_size,
                device="auto"
            )
        return self._stt
    
    def get_tts(self):
        """Get or create TTS model."""
        if self._tts is None:
            from src.tts.xtts import XTTSTS
            logger.info("Loading TTS model (once)...")
            self._tts = XTTSTS(device="auto")
            self._tts.load_model()  # Load model immediately
        return self._tts
    
    def get_prosody_extractor(self, sample_rate: int):
        """Get or create prosody extractor."""
        if self._prosody_extractor is None:
            from src.prosody import ProsodyExtractor
            logger.info("Loading prosody extractor (once)...")
            self._prosody_extractor = ProsodyExtractor(sample_rate=sample_rate)
        return self._prosody_extractor
    
    def get_speaker_model(self):
        """Get or create speaker model."""
        if self._speaker_model is None:
            from src.speaker import SpeakerEmbedding
            logger.info("Loading speaker model (once)...")
            self._speaker_model = SpeakerEmbedding(device="auto")
        return self._speaker_model
    
    def get_text_compressor(self, config=None):
        """Get or create text compressor.
        
        Args:
            config: Optional config to override self.config
        """
        # Always create new instance to support changing configs
        cfg = config if config is not None else self.config
        from src.compression.text import TextCompressor
        return TextCompressor(
            algorithm=cfg.compression.text_algorithm,
            level=cfg.compression.text_level,
            preprocess=cfg.compression.text_preprocess
        )
    
    def get_prosody_compressor(self, config=None):
        """Get or create prosody compressor.
        
        Args:
            config: Optional config to override self.config
        """
        # Always create new instance to support changing configs
        cfg = config if config is not None else self.config
        from src.compression.prosody import ProsodyCompressor
        return ProsodyCompressor(
            pitch_bits=cfg.compression.prosody_quantization_pitch_bits,
            energy_bits=cfg.compression.prosody_quantization_energy_bits,
            rate_bits=cfg.compression.prosody_quantization_rate_bits,
            input_rate=cfg.prosody.update_rate_hz,
            keyframe_rate=cfg.prosody.update_rate_hz
        )
    
    def get_timbre_compressor(self, config=None):
        """Get or create timbre compressor.
        
        Args:
            config: Optional config to override self.config
        """
        # Always create new instance to support changing configs
        cfg = config if config is not None else self.config
        from src.compression.timbre import TimbreCompressor
        return TimbreCompressor(
            algorithm=cfg.compression.timbre_algorithm
        )


def collect_librispeech_samples(num_samples: int) -> List[Dict[str, Any]]:
    """Collect LibriSpeech test samples.
    
    Args:
        num_samples: Number of samples to collect
        
    Returns:
        List of sample info dicts with 'speaker_id', 'chapter_id', 'flac_files', 'transcript_file'
    """
    librispeech_dir = Path("LibriSpeech/test-clean")
    
    if not librispeech_dir.exists():
        raise FileNotFoundError(f"LibriSpeech directory not found: {librispeech_dir}. Please download the test-clean dataset from the official website and put the decompressed LibriSpeech folder into the root dir of this project.")
    
    samples = []
    
    # Iterate through speaker directories
    for speaker_dir in sorted(librispeech_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
        
        speaker_id = speaker_dir.name
        
        # Iterate through chapter directories
        for chapter_dir in sorted(speaker_dir.iterdir()):
            if not chapter_dir.is_dir():
                continue
            
            chapter_id = chapter_dir.name
            
            # Find all .flac files
            flac_files = sorted(chapter_dir.glob("*.flac"))
            transcript_file = chapter_dir / f"{speaker_id}-{chapter_id}.trans.txt"
            
            if flac_files and transcript_file.exists():
                samples.append({
                    'speaker_id': speaker_id,
                    'chapter_id': chapter_id,
                    'flac_files': flac_files,
                    'transcript_file': transcript_file
                })
            
            if len(samples) >= num_samples:
                break
        
        if len(samples) >= num_samples:
            break
    
    logger.info(f"Collected {len(samples)} LibriSpeech samples")
    return samples[:num_samples]


def merge_flac_files(flac_files: List[Path]) -> Tuple[np.ndarray, int]:
    """Merge multiple FLAC files into a single audio array, up to 30 seconds.
    
    Args:
        flac_files: List of paths to FLAC files
        
    Returns:
        Tuple of (merged_audio, sample_rate)
    """
    import soundfile as sf
    import librosa
    
    audio_segments = []
    target_sr = 16000
    max_duration = 30.0  # seconds
    max_samples = int(max_duration * target_sr)
    current_samples = 0
    
    for flac_file in flac_files:
        audio, sr = sf.read(str(flac_file))
        
        # Resample to 16kHz if needed
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Check if adding this segment would exceed the limit
        if current_samples + len(audio) > max_samples:
            # Don't add this segment, stop merging
            break
        
        audio_segments.append(audio)
        current_samples += len(audio)
    
    # Concatenate all segments
    merged_audio = np.concatenate(audio_segments)
    
    return merged_audio, target_sr


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and Config objects."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        # Handle Config objects by converting to dict
        if hasattr(obj, '__dict__') and not isinstance(obj, type):
            # Recursively convert nested Config objects
            result = {}
            for key, value in obj.__dict__.items():
                if hasattr(value, '__dict__') and not isinstance(value, type):
                    result[key] = value.__dict__
                else:
                    result[key] = value
            return result
        return super().default(obj)


def get_baseline_cache_path(audio_path: str, baseline_name: str, bandwidth: float) -> Path:
    """Get cache file path for baseline results.
    
    Args:
        audio_path: Path to audio file
        baseline_name: Name of baseline (opus, encodec)
        bandwidth: Bandwidth/bitrate used
        
    Returns:
        Path to cache file
    """
    import hashlib
    
    # Create cache directory
    cache_dir = Path("benchmarks/.cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create unique cache key from audio path and parameters
    audio_path_obj = Path(audio_path)
    cache_key = f"{audio_path_obj.name}_{baseline_name}_{bandwidth:.1f}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    
    return cache_dir / f"{baseline_name}_{cache_hash}.json"


def save_baseline_cache(cache_path: Path, results: Dict[str, Any]):
    """Save baseline results to cache file.
    
    Args:
        cache_path: Path to cache file
        results: Results dictionary (without reconstructed_audio)
    """
    # Remove numpy arrays before saving
    cache_data = {k: v for k, v in results.items() if not isinstance(v, np.ndarray)}
    
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f, indent=2)
    
    logger.info(f"✅ Baseline results cached to: {cache_path}")


def load_baseline_cache(cache_path: Path) -> Dict[str, Any]:
    """Load baseline results from cache file.
    
    Args:
        cache_path: Path to cache file
        
    Returns:
        Results dictionary or None if cache doesn't exist
    """
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'r') as f:
            results = json.load(f)
        logger.info(f"📦 Loaded cached baseline results from: {cache_path}")
        return results
    except Exception as e:
        logger.warning(f"Failed to load cache: {e}")
        return None


def save_reconstructed_audio(
    audio: np.ndarray,
    sample_rate: int,
    original_audio_path: str,
    method_name: str
) -> Path:
    """Save reconstructed audio to file.
    
    Args:
        audio: Reconstructed audio array
        sample_rate: Sample rate
        original_audio_path: Path to original audio file
        method_name: Name of method/config (e.g., 'balanced_mode', 'opus')
        
    Returns:
        Path to saved file
    """
    import soundfile as sf
    from datetime import datetime
    
    # Create reconstructed directory next to original audio
    original_path = Path(original_audio_path)
    reconstructed_dir = original_path.parent / "reconstructed"
    reconstructed_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = original_path.stem
    output_filename = f"{base_name}_{method_name}_{timestamp}.wav"
    output_path = reconstructed_dir / output_filename
    
    # Save audio
    sf.write(output_path, audio, sample_rate)
    logger.info(f"💾 Saved reconstructed audio: {output_path}")
    
    return output_path


def load_audio(audio_path: str) -> tuple:
    """Load audio file.
    
    Returns:
        Tuple of (audio, sample_rate)
    """
    import soundfile as sf
    import librosa
    
    audio, sr = sf.read(audio_path)
    
    # Resample to 16kHz (standard for speech)
    if sr != 16000:
        logger.info(f"Resampling from {sr} Hz to 16000 Hz")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    return audio, sr


def benchmark_our_method(
    audio: np.ndarray,
    sample_rate: int,
    config: Any,
    reference_file: str = None,
    bit_error_rate: float = 0.0,
    model_manager: Optional[ModelManager] = None
) -> Dict[str, Any]:
    """Benchmark our STT-Compress-TTS method.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate
        config: Configuration object
        reference_file: Optional reference audio for voice cloning
        bit_error_rate: Bit error rate for noise injection (0.0 = no noise)
        model_manager: Optional ModelManager for reusing loaded models (speeds up batch processing)
    
    Returns:
        Dictionary with all metrics
    """
    logger.info("=" * 80)
    logger.info("BENCHMARKING: OUR METHOD")
    logger.info("=" * 80)
    
    results = {'method': 'ours'}
    timings = {}
    
    # Use model manager if provided, otherwise create models on-the-fly
    if model_manager:
        stt = model_manager.get_stt()
        prosody_extractor = model_manager.get_prosody_extractor(sample_rate)
        speaker_model = model_manager.get_speaker_model()
        text_compressor = model_manager.get_text_compressor(config)
        prosody_compressor = model_manager.get_prosody_compressor(config)
        timbre_compressor = model_manager.get_timbre_compressor(config)
        tts = model_manager.get_tts()
    else:
        from src.stt import FasterWhisperSTT
        from src.compression.text import TextCompressor
        from src.compression.prosody import ProsodyCompressor
        from src.compression.timbre import TimbreCompressor
        from src.prosody import ProsodyExtractor
        from src.speaker import SpeakerEmbedding
        from src.tts.xtts import XTTSTS
        
        stt = FasterWhisperSTT(
            model_size=config.stt.model_size,
            device="auto"
        )
        prosody_extractor = ProsodyExtractor(sample_rate=sample_rate)
        speaker_model = SpeakerEmbedding(device="auto")
        text_compressor = TextCompressor(
            algorithm=config.compression.text_algorithm,
            level=config.compression.text_level,
            preprocess=config.compression.text_preprocess
        )
        prosody_compressor = ProsodyCompressor(
            pitch_bits=config.compression.prosody_quantization_pitch_bits,
            energy_bits=config.compression.prosody_quantization_energy_bits,
            rate_bits=config.compression.prosody_quantization_rate_bits,
            input_rate=config.prosody.update_rate_hz,
            keyframe_rate=config.prosody.update_rate_hz
        )
        timbre_compressor = TimbreCompressor(algorithm=config.compression.timbre_algorithm)
        tts = XTTSTS(device="auto")
    
    # 1. STT - Transcribe
    logger.info("Step 1/5: Transcribing audio...")
    t0 = time.time()
    stt_result = stt.transcribe(audio, sample_rate)
    timings['stt'] = time.time() - t0
    text = stt_result.text
    results['original_text'] = text
    
    # 2. Compress text
    logger.info("Step 2/5: Compressing text...")
    t0 = time.time()
    compressed_text = text_compressor.compress(text)
    timings['text_compression'] = time.time() - t0
    text_bytes = len(compressed_text)
    results['text_bytes'] = text_bytes
    
    # 3. Extract and compress prosody (in chunks)
    logger.info("Step 3/5: Compressing prosody...")
    t0 = time.time()
    
    # Simulate streaming: chunk audio based on update rate
    update_rate = config.prosody.update_rate_hz
    chunk_duration = 1.0 / update_rate
    chunk_samples = int(sample_rate * chunk_duration)
    duration = len(audio) / sample_rate
    num_chunks = int(duration * update_rate)
    
    logger.info(f"Prosody extraction: {num_chunks} chunks @ {update_rate}Hz (chunk_duration={chunk_duration:.3f}s)")
    
    # Extract mean prosody values for all chunks first (for temporal delta encoding)
    pitch_values = []
    energy_values = []
    skipped_chunks = 0
    
    for i in range(num_chunks):
        start_idx = i * chunk_samples
        end_idx = min(start_idx + chunk_samples, len(audio))
        audio_chunk = audio[start_idx:end_idx]
        
        # Skip chunks that are too small (< 50ms minimum for prosody extraction)
        min_chunk_size = int(sample_rate * 0.05)  # 50ms minimum
        if len(audio_chunk) < min_chunk_size:
            skipped_chunks += 1
            continue
        
        # Extract prosody features from chunk
        prosody_chunk = prosody_extractor.extract_all(audio_chunk)
        
        # Extract mean values for temporal encoding
        if prosody_chunk.pitch is not None and len(prosody_chunk.pitch) > 0:
            pitch_values.append(np.mean(prosody_chunk.pitch))
        else:
            pitch_values.append(0.0)
            
        if prosody_chunk.energy is not None and len(prosody_chunk.energy) > 0:
            energy_values.append(np.mean(prosody_chunk.energy))
        else:
            energy_values.append(0.0)
    
    if skipped_chunks > 0:
        logger.warning(f"Skipped {skipped_chunks}/{num_chunks} chunks (too small for prosody extraction)")
    
    logger.info(f"Extracted {len(pitch_values)} prosody values (expected {num_chunks}, got {len(pitch_values)})")
    
    # Apply temporal delta encoding across packets if enabled
    if config.compression.prosody_delta_encoding and len(pitch_values) > 1:
        logger.info(f"Applying temporal delta encoding to {len(pitch_values)} prosody values")
        
        # Compress all pitch values together (enables delta encoding across time)
        pitch_array = np.array(pitch_values)
        compressed_pitch_all = prosody_compressor.compress_pitch(pitch_array)
        
        # Compress all energy values together
        energy_array = np.array(energy_values)
        compressed_energy_all = prosody_compressor.compress_energy(energy_array)
        
        total_prosody_bytes = len(compressed_pitch_all) + len(compressed_energy_all)
        
        # Store as single "packet" containing all temporal data
        compressed_prosody_packets = [{
            'pitch': compressed_pitch_all,
            'energy': compressed_energy_all,
            'pitch_length': len(pitch_values),
            'energy_length': len(energy_values)
        }]
        
        logger.info(f"Temporal encoding: {len(pitch_values)} values → {total_prosody_bytes}B "
                   f"({total_prosody_bytes/len(pitch_values):.1f}B/value)")
    else:
        # No temporal delta encoding - compress each value independently
        total_prosody_bytes = 0
        compressed_prosody_packets = []
        
        for pitch_val, energy_val in zip(pitch_values, energy_values):
            compressed_pitch = prosody_compressor.compress_pitch(np.array([pitch_val]))
            compressed_energy = prosody_compressor.compress_energy(np.array([energy_val]))
            
            compressed_prosody_packets.append({
                'pitch': compressed_pitch,
                'energy': compressed_energy,
                'pitch_length': 1,
                'energy_length': 1
            })
            
            total_prosody_bytes += len(compressed_pitch) + len(compressed_energy)
    
    results['prosody_bytes'] = total_prosody_bytes
    results['prosody_packets'] = len(compressed_prosody_packets)
    results['prosody_values'] = len(pitch_values)  # Actual number of prosody updates
    timings['prosody_processing'] = time.time() - t0
    
    # 4. Extract and compress speaker embedding
    logger.info("Step 4/5: Compressing speaker embedding...")
    t0 = time.time()
    embedding = speaker_model.extract(audio, sample_rate)
    compressed_timbre = timbre_compressor.compress(embedding)
    timings['timbre_processing'] = time.time() - t0
    timbre_bytes = len(compressed_timbre)
    results['timbre_bytes'] = timbre_bytes
    
    # Inject bit errors if noise testing is enabled
    if bit_error_rate > 0:
        from benchmarks.noise_injection import inject_errors_into_components
        
        logger.info(f"💥 INJECTING BIT ERRORS: BER = {bit_error_rate:.2e}")
        logger.info(f"   Target: Prosody and Timbre only (Text skipped - would cause decompression failure)")
        
        # Store original compressed data for reference
        results['original_text_bytes'] = text_bytes
        results['original_prosody_bytes'] = total_prosody_bytes
        results['original_timbre_bytes'] = timbre_bytes
        
        # Inject errors into prosody and timbre components only
        # Text is NOT corrupted because Brotli/LZ4 decompression fails with bit errors
        compressed_text, compressed_prosody_packets, compressed_timbre = inject_errors_into_components(
            compressed_text,
            compressed_prosody_packets,
            compressed_timbre,
            bit_error_rate,
            seed=42  # Fixed seed for reproducibility
        )
        
        # Recalculate sizes (should be same, but check for verification)
        text_bytes_after = len(compressed_text)
        timbre_bytes_after = len(compressed_timbre)
        total_prosody_bytes_after = sum(
            len(p.get('pitch', b'')) + len(p.get('energy', b'')) 
            for p in compressed_prosody_packets
        )
        
        logger.info(f"After noise injection:")
        logger.info(f"  Text: {text_bytes_after}B (unchanged: {text_bytes == text_bytes_after})")
        logger.info(f"  Prosody: {total_prosody_bytes_after}B (unchanged: {total_prosody_bytes == total_prosody_bytes_after})")
        logger.info(f"  Timbre: {timbre_bytes_after}B (unchanged: {timbre_bytes == timbre_bytes_after})")
        
        results['bit_error_rate'] = bit_error_rate
    else:
        results['bit_error_rate'] = 0.0
    
    # Calculate bitrate
    total_bytes = text_bytes + total_prosody_bytes + timbre_bytes
    duration_sec = duration
    bitrate = (total_bytes * 8) / duration_sec
    text_bps = (text_bytes * 8) / duration_sec
    prosody_bps = (total_prosody_bytes * 8) / duration_sec
    timbre_bps = (timbre_bytes * 8) / duration_sec
    
    results['total_bytes'] = total_bytes
    results['text_bytes'] = text_bytes
    results['duration'] = duration
    results['bitrate_bps'] = bitrate
    results['text_bps'] = text_bps
    results['prosody_bps'] = prosody_bps
    results['timbre_bps'] = timbre_bps
    
    logger.info(f"📊 Compression breakdown:")
    logger.info(f"  Text: {text_bytes}B ({text_bps:.1f} bps)")
    logger.info(f"  Prosody: {total_prosody_bytes}B ({prosody_bps:.1f} bps, {len(pitch_values)} updates)")
    logger.info(f"  Timbre: {timbre_bytes}B ({timbre_bps:.1f} bps)")
    logger.info(f"  Total: {total_bytes}B for {duration:.1f}s = {bitrate:.1f} bps")
    
    # 5. Reconstruct audio with TTS
    logger.info("Step 5/5: Reconstructing audio with TTS...")
    t0 = time.time()
    
    # First, decompress all prosody packets and reconstruct full contour
    logger.info(f"Decompressing {len(compressed_prosody_packets)} prosody packets...")
    decompressed_pitch_chunks = []
    decompressed_energy_chunks = []
    
    for packet in compressed_prosody_packets:
        # Decompress pitch
        if packet['pitch'] and packet['pitch_length'] > 0:
            decompressed_pitch = prosody_compressor.decompress_pitch(
                packet['pitch'], 
                length=packet['pitch_length']
            )
            decompressed_pitch_chunks.append(decompressed_pitch)
        
        # Decompress energy
        if packet['energy'] and packet['energy_length'] > 0:
            decompressed_energy = prosody_compressor.decompress_energy(
                packet['energy'], 
                length=packet['energy_length']
            )
            decompressed_energy_chunks.append(decompressed_energy)
    
    # Concatenate all chunks to reconstruct full prosody contour
    full_decompressed_pitch = np.concatenate(decompressed_pitch_chunks) if decompressed_pitch_chunks else None
    full_decompressed_energy = np.concatenate(decompressed_energy_chunks) if decompressed_energy_chunks else None
    
    reconstructed_prosody = {
        'pitch': full_decompressed_pitch,
        'energy': full_decompressed_energy,
        'speaking_rate': None
    }
    
    logger.info(f"Reconstructed prosody: pitch={len(full_decompressed_pitch) if full_decompressed_pitch is not None else 0} frames, "
                f"energy={len(full_decompressed_energy) if full_decompressed_energy is not None else 0} frames")
    
    try:
        decompressed_text = text_compressor.decompress(compressed_text)
        decompressed_embedding = timbre_compressor.decompress(compressed_timbre, dim=192)
        
        # Synthesize with reconstructed prosody
        logger.info("Synthesizing speech with decompressed prosody...")
        tts_result = tts.synthesize(
            text=decompressed_text,
            speaker_embedding=decompressed_embedding,
            prosody=reconstructed_prosody,  # Pass reconstructed prosody
            speaker_wav=reference_file or None
        )
        
        reconstructed_audio = tts_result.audio
        
        # Resample to match original
        if tts_result.sample_rate != sample_rate:
            import librosa
            reconstructed_audio = librosa.resample(
                reconstructed_audio,
                orig_sr=tts_result.sample_rate,
                target_sr=sample_rate
            )
        
        results['reconstructed_audio'] = reconstructed_audio
        results['reconstruction_success'] = True
        
        timings['receiver_processing'] = time.time() - t0
        
        # Calculate RTF and Latency
        sender_time = timings.get('stt', 0) + timings.get('text_compression', 0) + \
                      timings.get('prosody_processing', 0) + timings.get('timbre_processing', 0)
        receiver_time = timings.get('receiver_processing', 0)
        total_time = sender_time + receiver_time
        
        results['rtf'] = total_time / duration if duration > 0 else 0
        results['latency_sender'] = sender_time
        results['latency_receiver'] = receiver_time
        results['latency_total'] = total_time
        results['timings'] = timings
        
        logger.info(f"⏱️  TIMING ANALYSIS:")
        logger.info(f"   RTF: {results['rtf']:.4f}x (lower is faster)")
        logger.info(f"   Total Latency: {total_time*1000:.1f}ms")
        logger.info(f"   Sender: {sender_time*1000:.1f}ms (STT: {timings.get('stt', 0)*1000:.1f}ms)")
        logger.info(f"   Receiver: {receiver_time*1000:.1f}ms")
        
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        results['reconstruction_success'] = False
        results['reconstructed_audio'] = None
    
    return results


def benchmark_baseline(
    audio: np.ndarray,
    sample_rate: int,
    baseline_name: str,
    **kwargs
) -> Dict[str, Any]:
    """Benchmark a baseline method.
    
    Args:
        audio: Audio signal
        sample_rate: Sample rate
        baseline_name: 'opus' or 'encodec'
        **kwargs: Codec-specific parameters
        
    Returns:
        Dictionary with metrics
    """
    logger.info("=" * 80)
    logger.info(f"BENCHMARKING: {baseline_name.upper()}")
    logger.info("=" * 80)
    
    results = {'method': baseline_name}
    duration = len(audio) / sample_rate
    
    try:
        if baseline_name == 'opus':
            from benchmarks.baselines.opus import OpusCodec
            codec = OpusCodec(**kwargs)
            
            # Check if Opus is available
            if not codec.available:
                logger.error(f"❌ Opus codec not available - skipping baseline")
                logger.info("   Install with: pip install opuslib")
                results['reconstruction_success'] = False
                results['error'] = 'Opus codec not available'
                return results
                
        elif baseline_name == 'encodec':
            from benchmarks.baselines.encodec import EncodecCodec
            codec = EncodecCodec(**kwargs)
        else:
            logger.error(f"Unknown baseline: {baseline_name}")
            return results
        
        reconstructed_audio, compressed_size = codec.encode_decode(audio)
        
        # Check if encoding was successful
        if compressed_size == 0:
            logger.error(f"❌ {baseline_name.upper()} encoding failed")
            results['reconstruction_success'] = False
            results['error'] = f'{baseline_name} encoding failed'
            return results
        
        bitrate = (compressed_size * 8) / duration
        
        results['total_bytes'] = compressed_size
        results['duration'] = duration
        results['bitrate_bps'] = bitrate
        results['reconstructed_audio'] = reconstructed_audio
        results['reconstruction_success'] = True
        
        logger.info(f"Total: {compressed_size}B for {duration:.1f}s = {bitrate:.1f} bps")
        
    except Exception as e:
        logger.error(f"Baseline {baseline_name} failed: {e}")
        results['reconstruction_success'] = False
        results['reconstructed_audio'] = None
        results['bitrate_bps'] = 0
    
    return results


def calculate_metrics(
    original_audio: np.ndarray,
    reconstructed_audio: np.ndarray,
    sample_rate: int,
    original_text: str = None
) -> Dict[str, Any]:
    """Calculate all evaluation metrics.
    
    Returns:
        Dictionary with all metric scores
    """
    from benchmarks.metrics.wer import calculate_wer_from_audio
    from benchmarks.metrics.speaker_similarity import calculate_speaker_similarity
    from benchmarks.metrics.perceptual_quality import (
        calculate_pesq,
        calculate_stoi,
        calculate_nisqa
    )
    
    logger.info("=" * 80)
    logger.info("CALCULATING METRICS")
    logger.info("=" * 80)
    
    metrics = {}
    
    # Match lengths
    min_len = min(len(original_audio), len(reconstructed_audio))
    original_audio = original_audio[:min_len]
    reconstructed_audio = reconstructed_audio[:min_len]
    
    # 1. WER (Word Error Rate)
    logger.info("1/5: Calculating WER...")
    try:
        wer_results = calculate_wer_from_audio(original_audio, reconstructed_audio, sample_rate)
        metrics['wer'] = wer_results['wer']
        metrics['reconstructed_text'] = wer_results['reconstructed_text']
    except Exception as e:
        logger.error(f"WER calculation failed: {e}")
        metrics['wer'] = None
    
    # 2. Speaker Similarity
    logger.info("2/5: Calculating Speaker Similarity...")
    try:
        speaker_results = calculate_speaker_similarity(original_audio, reconstructed_audio, sample_rate)
        metrics['speaker_similarity'] = speaker_results['speaker_similarity']
        metrics['is_same_speaker'] = speaker_results['is_same_speaker']
    except Exception as e:
        logger.error(f"Speaker similarity calculation failed: {e}")
        metrics['speaker_similarity'] = None
    
    # 3. PESQ
    logger.info("3/5: Calculating PESQ...")
    try:
        metrics['pesq'] = calculate_pesq(original_audio, reconstructed_audio, sample_rate)
    except Exception as e:
        logger.error(f"PESQ calculation failed: {e}")
        metrics['pesq'] = None
    
    # 4. STOI
    logger.info("4/5: Calculating STOI...")
    try:
        metrics['stoi'] = calculate_stoi(original_audio, reconstructed_audio, sample_rate)
    except Exception as e:
        logger.error(f"STOI calculation failed: {e}")
        metrics['stoi'] = None
    
    # 5. NISQA (Non-Intrusive Speech Quality Assessment)
    logger.info("5/5: Calculating NISQA...")
    try:
        nisqa_results = calculate_nisqa(reconstructed_audio, sample_rate)
        metrics['nisqa_mos'] = nisqa_results.get('mos', None)
        metrics['nisqa_noisiness'] = nisqa_results.get('noisiness', None)
        metrics['nisqa_coloration'] = nisqa_results.get('coloration', None)
        metrics['nisqa_discontinuity'] = nisqa_results.get('discontinuity', None)
        metrics['nisqa_loudness'] = nisqa_results.get('loudness', None)
    except Exception as e:
        logger.error(f"NISQA calculation failed: {e}")
        metrics['nisqa_mos'] = None
    
    return metrics


def print_results_table(all_results: list):
    """Print formatted results table."""
    logger.info("=" * 80)
    logger.info("BENCHMARK RESULTS SUMMARY")
    logger.info("=" * 80)
    
    # Check if noise testing was performed
    has_noise_tests = any(r.get('bit_error_rate', 0) > 0 for r in all_results)
    
    # Print bitrate comparison
    print("\n📊 BITRATE COMPARISON")
    if has_noise_tests:
        print("-" * 140)
        print(f"{'Method':<25} {'Noise':<15} {'Total (bps)':<12} {'Text (bps)':<12} {'Prosody (bps)':<14} {'Timbre (bps)':<14} {'Within Target':<15}")
        print("-" * 140)
    else:
        print("-" * 120)
        print(f"{'Method':<25} {'Total (bps)':<12} {'Text (bps)':<12} {'Prosody (bps)':<14} {'Timbre (bps)':<14} {'Within Target':<15}")
        print("-" * 120)
    
    target_bps = 650
    for result in all_results:
        # Create method name with config if available
        method = result['method']
        if 'config' in result:
            method = f"{method} ({result['config']})"
        
        bitrate = result.get('bitrate_bps', 0)
        text_bps = result.get('text_bps', 0)
        prosody_bps = result.get('prosody_bps', 0)
        timbre_bps = result.get('timbre_bps', 0)
        within_target = "✅ YES" if bitrate <= target_bps else "❌ NO"
        
        # Format with proper fallback for baselines
        text_str = f"{text_bps:.1f}" if text_bps > 0 else "N/A"
        prosody_str = f"{prosody_bps:.1f}" if prosody_bps > 0 else "N/A"
        timbre_str = f"{timbre_bps:.1f}" if timbre_bps > 0 else "N/A"
        
        if has_noise_tests:
            noise_level = result.get('noise_level', 'No noise')
            print(f"{method:<25} {noise_level:<15} {bitrate:<12.1f} {text_str:<12} {prosody_str:<14} {timbre_str:<14} {within_target:<15}")
        else:
            print(f"{method:<25} {bitrate:<12.1f} {text_str:<12} {prosody_str:<14} {timbre_str:<14} {within_target:<15}")
    
    # Print quality metrics
    print("\n🎯 QUALITY METRICS")
    if has_noise_tests:
        print("-" * 100)
        print(f"{'Method':<25} {'Noise':<15} {'WER':<10} {'SpkrSim':<10} {'PESQ':<10} {'STOI':<10} {'NISQA':<10}")
        print("-" * 100)
    else:
        print("-" * 80)
        print(f"{'Method':<25} {'WER':<10} {'SpkrSim':<10} {'PESQ':<10} {'STOI':<10} {'NISQA':<10}")
        print("-" * 80)
    
    for result in all_results:
        # Create method name with config if available
        method = result['method']
        if 'config' in result:
            method = f"{method} ({result['config']})"
        
        wer = result.get('wer', None)
        spkr = result.get('speaker_similarity', None)
        pesq = result.get('pesq', None)
        stoi = result.get('stoi', None)
        nisqa = result.get('nisqa_mos', None)
        
        wer_str = f"{wer:.3f}" if wer is not None else "N/A"
        spkr_str = f"{spkr:.3f}" if spkr is not None else "N/A"
        pesq_str = f"{pesq:.2f}" if pesq is not None else "N/A"
        stoi_str = f"{stoi:.3f}" if stoi is not None else "N/A"
        nisqa_str = f"{nisqa:.3f}" if nisqa is not None else "N/A"
        
        if has_noise_tests:
            noise_level = result.get('noise_level', 'No noise')
            print(f"{method:<25} {noise_level:<15} {wer_str:<10} {spkr_str:<10} {pesq_str:<10} {stoi_str:<10} {nisqa_str:<10}")
        else:
            print(f"{method:<25} {wer_str:<10} {spkr_str:<10} {pesq_str:<10} {stoi_str:<10} {nisqa_str:<10}")
    
    print("\n" + "=" * 80)
    print("✅ Lower WER = Better intelligibility")
    print("✅ Higher Speaker Similarity = Better voice preservation")
    print("✅ Higher PESQ/STOI/NISQA = Better perceptual quality")
    print("   NISQA MOS range: 1-5 (5 = excellent)")
    print("=" * 80)


def interpret_results(json_path: Path) -> None:
    """Interpret and summarize benchmark results from JSON file.
    
    Args:
        json_path: Path to JSON results file
    """
    import pandas as pd
    
    logger.info("=" * 80)
    logger.info("📊 INTERPRETING BENCHMARK RESULTS")
    logger.info("=" * 80)
    
    if not json_path.exists():
        logger.error(f"❌ JSON file not found: {json_path}")
        return
    
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    logger.info(f"Loaded {len(results)} results from {json_path}")
    
    # Separate average results from individual samples
    avg_results = [r for r in results if r.get('is_average')]
    sample_results = [r for r in results if not r.get('is_average')]
    
    logger.info(f"  - {len(sample_results)} individual sample results")
    logger.info(f"  - {len(avg_results)} averaged statistics")
    
    if not avg_results:
        logger.warning("⚠️  No averaged results found in JSON file")
        return
    
    # Fill in missing statistics by recalculating from sample data if needed
    for avg_result in avg_results:
        # Determine the grouping key (method or config)
        group_key = 'method' if 'method' in avg_result else 'config'
        group_value = avg_result.get(group_key)
        noise_level = avg_result.get('noise_level')
        
        # Get matching samples
        matching_samples = [s for s in sample_results 
                          if s.get(group_key) == group_value 
                          and (noise_level is None or s.get('noise_level') == noise_level)]
        
        if matching_samples:
            # List of numeric fields that should have statistics
            numeric_fields = ['speaker_similarity', 'wer', 'bitrate_bps', 'text_bps', 
                            'prosody_bps', 'timbre_bps', 'pesq', 'stoi', 'nisqa_mos',
                            'rtf', 'latency_total', 'latency_sender', 'latency_receiver']
            
            for field in numeric_fields:
                # If the average doesn't have this field, calculate it from samples
                if field not in avg_result:
                    values = [s[field] for s in matching_samples if field in s and s[field] is not None]
                    if field == 'rtf' and len(values) > 1:
                        values = values[1:]
                    if values:
                        avg_result[field] = np.mean(values)
                        avg_result[f'{field}_std'] = np.std(values)
    
    # Create DataFrame for easier analysis
    df = pd.DataFrame(avg_results)
    
    # Detect if this is baseline results (has 'method' but no 'config')
    # or our method results (has 'config')
    is_baseline = 'method' in df.columns and 'config' not in df.columns
    config_key = 'method' if is_baseline else 'config'
    
    # Group results by config/method and noise level (if exists)
    configs = df[config_key].unique()
    
    logger.info("\n" + "=" * 80)
    logger.info("📈 SUMMARY STATISTICS")
    logger.info("=" * 80)
    
    # Metrics to display
    metrics = {
        'bitrate_bps': ('Total Bitrate', 'bps'),
        'text_bps': ('Text', 'bps'),
        'prosody_bps': ('Prosody', 'bps'),
        'timbre_bps': ('Timbre', 'bps'),
        'wer': ('WER', ''),
        'speaker_similarity': ('Speaker Sim', ''),
        'pesq': ('PESQ', ''),
        'stoi': ('STOI', ''),
        'nisqa_mos': ('NISQA MOS', ''),
        'rtf': ('RTF', 'x'),
        'latency_total': ('Latency', 's'),
    }
    
    for config in configs:
        config_data = df[df[config_key] == config]
        
        logger.info(f"\n{'─' * 80}")
        logger.info(f"{config_key.upper()}: {config}")
        logger.info(f"{'─' * 80}")
        
        # Check if noise_level exists in data
        has_noise_levels = 'noise_level' in config_data.columns and not config_data['noise_level'].isna().all()
        
        if has_noise_levels:
            # Get unique noise levels for this config
            noise_levels = config_data['noise_level'].unique()
            
            for noise_level in noise_levels:
                row = config_data[config_data['noise_level'] == noise_level].iloc[0]
                num_samples = row.get('num_samples', 0)
                
                logger.info(f"\n  Noise Level: {noise_level} ({num_samples} samples)")
                logger.info(f"  {'─' * 76}")
                
                # Display compression metrics
                logger.info(f"\n  📦 Compression:")
                for key, (name, unit) in [('bitrate_bps', metrics['bitrate_bps']), 
                                          ('text_bps', metrics['text_bps']),
                                          ('prosody_bps', metrics['prosody_bps']),
                                          ('timbre_bps', metrics['timbre_bps'])]:
                    if key in row:
                        mean_val = row[key]
                        std_val = row.get(f'{key}_std', 0)
                        logger.info(f"    {name:15s}: {mean_val:7.1f} ± {std_val:5.1f} {unit}")
                
                # Display latency metrics
                logger.info(f"\n  ⏱️  Latency:")
                for key, (name, unit) in [('rtf', metrics['rtf']), 
                                          ('latency_total', metrics['latency_total'])]:
                    if key in row:
                        mean_val = row[key]
                        std_val = row.get(f'{key}_std', 0)
                        logger.info(f"    {name:15s}: {mean_val:7.4f} ± {std_val:6.4f} {unit}")

                # Display quality metrics
                logger.info(f"\n  🎯 Quality Metrics:")
                for key, (name, unit) in [('wer', metrics['wer']),
                                          ('speaker_similarity', metrics['speaker_similarity']),
                                          ('pesq', metrics['pesq']),
                                          ('stoi', metrics['stoi']),
                                          ('nisqa_mos', metrics['nisqa_mos'])]:
                    if key in row:
                        mean_val = row[key]
                        std_val = row.get(f'{key}_std', 0)
                        logger.info(f"    {name:15s}: {mean_val:7.4f} ± {std_val:6.4f} {unit}")
        else:
            # No noise levels (baseline results)
            row = config_data.iloc[0]
            num_samples = row.get('num_samples', 0)
            
            logger.info(f"\n  Samples: {num_samples}")
            logger.info(f"  {'─' * 76}")
            
            # Display compression metrics
            logger.info(f"\n  📦 Compression:")
            if 'bitrate_bps' in row:
                mean_val = row['bitrate_bps']
                std_val = row.get('bitrate_bps_std', 0)
                logger.info(f"    {'Total Bitrate':15s}: {mean_val:7.1f} ± {std_val:5.1f} bps")
            
            # Display quality metrics
            logger.info(f"\n  🎯 Quality Metrics:")
            for key, (name, unit) in [('pesq', metrics['pesq']),
                                      ('stoi', metrics['stoi']),
                                      ('nisqa_mos', metrics['nisqa_mos'])]:
                if key in row:
                    mean_val = row[key]
                    std_val = row.get(f'{key}_std', 0)
                    logger.info(f"    {name:15s}: {mean_val:7.4f} ± {std_val:6.4f} {unit}")
    
    # Generate comparison table
    logger.info("\n" + "=" * 80)
    logger.info("📊 COMPARISON TABLE")
    logger.info("=" * 80)
    
    # Check if this is baseline or our method results
    has_noise_levels = 'noise_level' in df.columns and not df['noise_level'].isna().all()
    
    if is_baseline:
        # Simplified table for baselines (no text/prosody/timbre breakdown)
        header = f"\n{'Method':<20s} {'Total (bps)':<15s} {'WER':<15s} {'Spkr Sim':<15s} {'PESQ':<15s} {'STOI':<15s} {'NISQA MOS':<15s}"
        logger.info(header)
        logger.info("─" * 110)
        
        for config in configs:
            config_data = df[df[config_key] == config]
            row = config_data.iloc[0]
            
            config_str = config[:19]
            
            # Format with mean±std
            def format_val(key, decimals=1):
                if key in row:
                    mean = row[key]
                    std = row.get(f'{key}_std', 0)
                    return f"{mean:.{decimals}f}±{std:.{decimals}f}"
                return "N/A"
            
            total_str = format_val('bitrate_bps', 1)
            wer_str = format_val('wer', 4)
            spkr_str = format_val('speaker_similarity', 4)
            pesq_str = format_val('pesq', 4)
            stoi_str = format_val('stoi', 4)
            nisqa_str = format_val('nisqa_mos', 4)
            
            line = f"{config_str:<20s} {total_str:<15s} {wer_str:<15s} {spkr_str:<15s} {pesq_str:<15s} {stoi_str:<15s} {nisqa_str:<15s}"
            logger.info(line)
    else:
        # Full table for our method with bitrate breakdown
        header = f"\n{'Config':<20s} {'Noise':<12s} {'Total (bps)':<15s} {'w/o Tmb (bps)':<15s} {'Text (bps)':<13s} {'Pros (bps)':<13s} {'Timb (bps)':<13s} {'RTF':<10s} {'WER':<13s} {'Spkr Sim':<13s} {'PESQ':<13s} {'STOI':<13s} {'NISQA MOS':<13s}"
        logger.info(header)
        logger.info("─" * 175)
        
        # Print each row
        for config in configs:
            config_data = df[df[config_key] == config]
            
            if has_noise_levels:
                noise_levels = config_data['noise_level'].unique()
            else:
                noise_levels = ['N/A']
            
            for noise_level in noise_levels:
                if has_noise_levels:
                    row = config_data[config_data['noise_level'] == noise_level].iloc[0]
                else:
                    row = config_data.iloc[0]
                
                config_str = config[:19]
                noise_str = str(noise_level)[:11] if has_noise_levels else "N/A"
                
                # Format with mean±std
                def format_val(key):
                    if key in row:
                        mean = row[key]
                        std = row.get(f'{key}_std', 0)
                        return f"{mean:.1f}±{std:.1f}"
                    return "N/A"
                
                def format_metric(key, decimals=4):
                    if key in row:
                        mean = row[key]
                        std = row.get(f'{key}_std', 0)
                        return f"{mean:.{decimals}f}±{std:.{decimals}f}"
                    return "N/A"
                
                total_str = format_val('bitrate_bps')
                text_bps = row.get('text_bps', 0) if 'text_bps' in row else 0
                text_std = row.get('text_bps_std', 0)
                pros_bps = row.get('prosody_bps', 0) if 'prosody_bps' in row else 0
                pros_std = row.get('prosody_bps_std', 0)
                # Calculate w/o timbre with propagated uncertainty
                wo_timbre_mean = text_bps + pros_bps
                wo_timbre_std = (text_std**2 + pros_std**2)**0.5  # Error propagation
                wo_timbre_str = f"{wo_timbre_mean:.1f}±{wo_timbre_std:.1f}"
                text_str = format_val('text_bps')
                pros_str = format_val('prosody_bps')
                timb_str = format_val('timbre_bps')
                rtf_str = format_metric('rtf', 3)
                wer_str = format_metric('wer', 4)
                spkr_str = format_metric('speaker_similarity', 4)
                pesq_str = format_metric('pesq', 4)
                stoi_str = format_metric('stoi', 4)
                nisqa_str = format_metric('nisqa_mos', 4)
                
                line = f"{config_str:<20s} {noise_str:<12s} {total_str:<15s} {wo_timbre_str:<15s} {text_str:<13s} {pros_str:<13s} {timb_str:<13s} {rtf_str:<10s} {wer_str:<13s} {spkr_str:<13s} {pesq_str:<13s} {stoi_str:<13s} {nisqa_str:<13s}"
                logger.info(line)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ INTERPRETATION COMPLETE")
    logger.info("=" * 80)


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(description="Run comprehensive benchmarks")
    parser.add_argument("--config", "-c", help="Config file (e.g., minimal_mode). If not specified, runs all configs in configs/")
    parser.add_argument("--audio", "-a", help="Audio file to test")
    parser.add_argument("--reference", "-r", help="Reference audio for TTS voice cloning (default: same as --audio)")
    parser.add_argument("--baselines", "-b", nargs="+", default=["opus", "encodec"],
                       help="Baselines to compare (opus, encodec)")
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument("--skip-metrics", action="store_true", help="Skip metric calculation")
    parser.add_argument("--baseline-only", action="store_true", help="Only run baseline benchmarks (skip our method)")
    parser.add_argument("--skip-baseline-cache", action="store_true", help="Force re-run baseline benchmarks (ignore cached results)")
    parser.add_argument("--plot", action="store_true", help="Run plot configs (plot_*_mode.yaml) for parameter sweep analysis")
    parser.add_argument("--plot-using-json", help="Generate plots from saved JSON results without reprocessing")
    parser.add_argument("--noise", metavar="CONFIG", help="Enable noise resilience testing using specified config as base (e.g., --noise high_quality). Tests bit error rates: 0.1%%, 1%%, 10%%")
    parser.add_argument("--librispeech", type=int, metavar="NUM", help="Run benchmark on NUM LibriSpeech test samples (optimized with model caching)")
    parser.add_argument("--interpret", help="Interpret and summarize results from JSON file")
    
    args = parser.parse_args()
    
    # If interpreting results, do that and exit
    if args.interpret:
        interpret_results(Path(args.interpret))
        return
    
    # Store noise config separately to avoid interference with normal config processing
    noise_config = args.noise
    if noise_config:
        logger.info("🔬 Noise resilience testing enabled")
        logger.info(f"   Noise config: {noise_config}")
        logger.info("   Bit error rates: 0.1%, 1%, 10%")
    
    # If plotting from JSON, do that and exit
    if args.plot_using_json:
        from benchmarks.plot_results import generate_all_plots
        import numpy as np
        
        logger.info("=" * 80)
        logger.info("📊 GENERATING PLOTS FROM JSON")
        logger.info("=" * 80)
        
        json_path = Path(args.plot_using_json)
        if not json_path.exists():
            logger.error(f"❌ JSON file not found: {json_path}")
            return
        
        with open(json_path, 'r') as f:
            results = json.load(f)
        
        logger.info(f"Loaded {len(results)} results from {json_path}")
        
        # Filter out existing averages to avoid double counting if we are recalculating
        # But if the file ONLY contains averages (e.g. from --librispeech run), we should use them.
        # The user asked to "calculate the mean value", implying we might have raw samples.
        
        raw_samples = [r for r in results if not r.get('is_average', False)]
        existing_averages = [r for r in results if r.get('is_average', False)]
        
        if not raw_samples and existing_averages:
            logger.info(f"Found {len(existing_averages)} existing averaged results. Using them directly.")
            aggregated_results = existing_averages
        elif not raw_samples and not existing_averages:
            logger.warning("No results found in JSON.")
            return
        else:
            logger.info(f"Found {len(raw_samples)} raw samples. Calculating means...")
            
            # Group by (method, config, noise_level)
            groups = {}
            for r in raw_samples:
                method = r.get('method', 'unknown')
                config = r.get('config', None)
                noise = r.get('noise_level', 'No noise')
                
                key = (method, config, noise)
                if key not in groups:
                    groups[key] = []
                groups[key].append(r)
            
            aggregated_results = []
            numeric_keys = ['bitrate_bps', 'text_bps', 'prosody_bps', 'timbre_bps', 
                           'wer', 'speaker_similarity', 'pesq', 'stoi',
                           'nisqa_mos', 'nisqa_noisiness', 'nisqa_coloration', 
                           'nisqa_discontinuity', 'nisqa_loudness', 'duration',
                           'rtf', 'latency_total', 'latency_sender', 'latency_receiver']
            
            for key, samples in groups.items():
                method, config_name, noise_level = key
                
                # Base averaged result
                avg_result = {
                    'method': method,
                    'config': config_name,
                    'noise_level': noise_level,
                    'is_average': True,
                    'num_samples': len(samples),
                    'reconstruction_success': True  # Assume success for averaged results if they exist
                }
                
                # Copy config_data from first sample if available
                if samples[0].get('config_data'):
                    avg_result['config_data'] = samples[0]['config_data']
                
                # Calculate means and stds
                for field in numeric_keys:
                    values = [s[field] for s in samples if field in s and s[field] is not None]
                    if field == 'rtf' and len(values) > 1:
                        # Sometimes first RTF is outlier due to model loading, but here we just take all
                        pass
                        
                    if values:
                        avg_result[field] = float(np.mean(values))
                        avg_result[f'{field}_std'] = float(np.std(values))
                
                aggregated_results.append(avg_result)
            
            logger.info(f"Calculated {len(aggregated_results)} averaged data points.")

        # Determine output directory
        plots_dir = json_path.parent / "plots"
        
        try:
            generate_all_plots(aggregated_results, plots_dir)
            logger.info("=" * 80)
            logger.info("✅ PLOT GENERATION COMPLETE!")
            logger.info(f"📁 View your beautiful plots in: {plots_dir.absolute()}")
            logger.info("=" * 80)
        except Exception as e:
            logger.error(f"❌ Error generating plots: {e}")
            import traceback
            traceback.print_exc()
        
        return
    
    # Validate required arguments for normal operation
    if not args.audio and not args.librispeech:
        parser.error("Either --audio or --librispeech is required (unless using --plot-using-json)")
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Handle LibriSpeech batch processing
    if args.librispeech:
        logger.info("=" * 80)
        logger.info(f"📚 LIBRISPEECH BATCH BENCHMARK MODE")
        logger.info(f"   Processing {args.librispeech} samples")
        if args.baseline_only:
            logger.info(f"   Mode: Baseline-only ({', '.join(args.baselines)})")
        logger.info("=" * 80)
        
        # Collect LibriSpeech samples
        try:
            librispeech_samples = collect_librispeech_samples(args.librispeech)
        except FileNotFoundError as e:
            logger.error(f"❌ {e}")
            return
        
        all_results = []  # Initialize for LibriSpeech mode
        
        # Handle baseline-only mode
        if args.baseline_only:
            logger.info("=" * 80)
            logger.info("📊 BASELINE BENCHMARKING MODE")
            logger.info(f"   Testing baselines: {args.baselines}")
            logger.info("=" * 80)
            
            # Process each baseline
            for baseline_name in args.baselines:
                logger.info("=" * 80)
                logger.info(f"BASELINE: {baseline_name.upper()}")
                logger.info("=" * 80)
                
                sample_results = []
                
                # Process each LibriSpeech sample
                for idx, sample_info in enumerate(librispeech_samples, 1):
                    speaker_id = sample_info['speaker_id']
                    chapter_id = sample_info['chapter_id']
                    
                    logger.info("=" * 60)
                    logger.info(f"📖 Sample {idx}/{len(librispeech_samples)}: Speaker {speaker_id}, Chapter {chapter_id}")
                    logger.info("=" * 60)
                    
                    # Merge FLAC files into single audio
                    logger.info(f"Merging {len(sample_info['flac_files'])} FLAC files...")
                    audio, sample_rate = merge_flac_files(sample_info['flac_files'])
                    logger.info(f"Merged audio: {len(audio)/sample_rate:.1f}s @ {sample_rate} Hz")
                    
                    # Benchmark baseline
                    baseline_results = benchmark_baseline(
                        audio, sample_rate, baseline_name
                    )
                    
                    # Add metadata
                    baseline_results['method'] = baseline_name
                    baseline_results['librispeech_speaker'] = speaker_id
                    baseline_results['librispeech_chapter'] = chapter_id
                    baseline_results['sample_index'] = idx
                    
                    # Calculate metrics if needed
                    if not args.skip_metrics and baseline_results.get('reconstruction_success'):
                        metrics = calculate_metrics(
                            audio,
                            baseline_results['reconstructed_audio'],
                            sample_rate,
                            None  # No original text for baselines
                        )
                        baseline_results.update(metrics)
                    
                    sample_results.append(baseline_results)
                
                # Calculate averaged statistics for this baseline
                logger.info("=" * 80)
                logger.info(f"📊 AVERAGE STATISTICS for {baseline_name}")
                logger.info("=" * 80)
                
                # Calculate averages for numeric metrics
                avg_results = {'method': baseline_name, 'is_average': True}
                numeric_keys = ['bitrate_bps', 'speaker_similarity', 'pesq', 'stoi',
                               'nisqa_mos', 'nisqa_noisiness', 'nisqa_coloration', 
                               'nisqa_discontinuity', 'nisqa_loudness', 'duration']
                
                for key in numeric_keys:
                    values = [r[key] for r in sample_results if key in r and r[key] is not None]
                    if values:
                        avg_results[key] = np.mean(values)
                        avg_results[f'{key}_std'] = np.std(values)
                
                avg_results['num_samples'] = len(sample_results)
                
                # Log averaged results
                logger.info(f"Average ({len(sample_results)} samples):")
                logger.info(f"  Bitrate: {avg_results.get('bitrate_bps', 0):.1f} ± {avg_results.get('bitrate_bps_std', 0):.1f} bps")
                if 'nisqa_mos' in avg_results:
                    logger.info(f"  NISQA MOS: {avg_results.get('nisqa_mos', 0):.3f} ± {avg_results.get('nisqa_mos_std', 0):.3f}")
                if 'pesq' in avg_results:
                    logger.info(f"  PESQ: {avg_results.get('pesq', 0):.3f} ± {avg_results.get('pesq_std', 0):.3f}")
                
                sample_results.append(avg_results)
                all_results.extend(sample_results)
            
            # Save results
            if args.output:
                # Remove numpy arrays before JSON serialization
                for result in all_results:
                    if 'reconstructed_audio' in result:
                        del result['reconstructed_audio']
                
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w') as f:
                    json.dump(all_results, f, indent=2, cls=NumpyEncoder)
                
                logger.info(f"✅ Results saved to: {output_path}")
            
            logger.info("=" * 80)
            logger.info("✅ LIBRISPEECH BASELINE BENCHMARK COMPLETE!")
            logger.info("=" * 80)
            
            return
        
        # Determine which configs to run for normal benchmarking
        if args.plot:
            # Plot mode: run all configs in plot_configs/
            configs_dir = Path("plot_configs")
            config_files = list(configs_dir.glob("*.yaml"))
            normal_configs = [f.stem for f in config_files]
            logger.info(f"Plot mode enabled. Running all plot configs: {normal_configs}")
        elif args.config and not noise_config:
            # User specified a config without noise testing
            normal_configs = [args.config]
        elif not noise_config:
            # No config specified and no noise testing - run all configs
            configs_dir = Path("configs")
            config_files = list(configs_dir.glob("*_mode.yaml"))
            normal_configs = [f.stem for f in config_files if not f.stem.startswith("plot_")]
            logger.info(f"No config specified, running all configs: {normal_configs}")
        else:
            # Noise testing specified - run all configs for normal, then noise config separately
            if args.config:
                # User specified a config - only run that for normal benchmarking
                normal_configs = [args.config]
            else:
                # Run all configs for normal benchmarking
                configs_dir = Path("configs")
                config_files = list(configs_dir.glob("*_mode.yaml"))
                normal_configs = [f.stem for f in config_files if not f.stem.startswith("plot_")]
                logger.info(f"Running all configs for normal benchmarking: {normal_configs}")
        
        all_results = []  # Initialize for LibriSpeech mode
        
        # Process normal configs (without noise)
        for config_name in normal_configs:
            logger.info("=" * 80)
            logger.info(f"CONFIG: {config_name} (No noise)")
            logger.info("=" * 80)
            
            # Load config
            if args.plot:
                config_path = Path("plot_configs") / f"{config_name}.yaml"
                config = load_config(config_path)
            else:
                config = load_config(config_name)
            
            # Create model manager for efficient batch processing
            logger.info("🚀 Initializing models (one-time load for all samples)...")
            model_manager = ModelManager(config)
            
            # No noise for normal configs
            ber_levels = [(0.0, "No noise")]
            
            # Process each LibriSpeech sample
            sample_results = []
            
            for idx, sample_info in enumerate(librispeech_samples, 1):
                speaker_id = sample_info['speaker_id']
                chapter_id = sample_info['chapter_id']
                
                logger.info("=" * 60)
                logger.info(f"📖 Sample {idx}/{len(librispeech_samples)}: Speaker {speaker_id}, Chapter {chapter_id}")
                logger.info("=" * 60)
                
                # Merge FLAC files into single audio
                logger.info(f"Merging {len(sample_info['flac_files'])} FLAC files...")
                audio, sample_rate = merge_flac_files(sample_info['flac_files'])
                logger.info(f"Merged audio: {len(audio)/sample_rate:.1f}s @ {sample_rate} Hz")
                
                # Use first FLAC file as reference for voice cloning
                reference_file = str(sample_info['flac_files'][0])
                
                # Benchmark our method with model_manager
                our_results = benchmark_our_method(
                    audio, sample_rate, config, reference_file, 
                    bit_error_rate=0.0, model_manager=model_manager
                )
                
                # Add metadata
                our_results['config'] = config_name
                our_results['config_data'] = config
                our_results['noise_level'] = "No noise"
                our_results['librispeech_speaker'] = speaker_id
                our_results['librispeech_chapter'] = chapter_id
                our_results['sample_index'] = idx
                
                # Calculate metrics if needed
                if not args.skip_metrics and our_results.get('reconstruction_success'):
                    metrics = calculate_metrics(
                        audio,
                        our_results['reconstructed_audio'],
                        sample_rate,
                        our_results.get('original_text')
                    )
                    our_results.update(metrics)
                
                sample_results.append(our_results)
            
            # Calculate averaged statistics for this config
            logger.info("=" * 80)
            logger.info(f"📊 AVERAGE STATISTICS for {config_name}")
            logger.info("=" * 80)
            
            # Calculate averages for numeric metrics
            avg_results = {'method': 'ours', 'config': config_name, 'noise_level': "No noise", 'is_average': True}
            numeric_keys = ['bitrate_bps', 'text_bps', 'prosody_bps', 'timbre_bps', 
                           'wer', 'speaker_similarity', 'pesq', 'stoi',
                           'nisqa_mos', 'nisqa_noisiness', 'nisqa_coloration', 
                           'nisqa_discontinuity', 'nisqa_loudness', 'duration',
                           'rtf', 'latency_total', 'latency_sender', 'latency_receiver']
            
            for key in numeric_keys:
                values = [r[key] for r in sample_results if key in r and r[key] is not None]
                if key == 'rtf' and len(values) > 1:
                    values = values[1:]
                if values:
                    avg_results[key] = np.mean(values)
                    avg_results[f'{key}_std'] = np.std(values)
            
            avg_results['num_samples'] = len(sample_results)
            avg_results['config_data'] = config
            
            # Log averaged results
            logger.info(f"Average ({len(sample_results)} samples):")
            logger.info(f"  Bitrate: {avg_results.get('bitrate_bps', 0):.1f} ± {avg_results.get('bitrate_bps_std', 0):.1f} bps")
            if 'nisqa_mos' in avg_results:
                logger.info(f"  NISQA MOS: {avg_results.get('nisqa_mos', 0):.3f} ± {avg_results.get('nisqa_mos_std', 0):.3f}")
            if 'wer' in avg_results:
                logger.info(f"  WER: {avg_results.get('wer', 0):.3f} ± {avg_results.get('wer_std', 0):.3f}")
            
            sample_results.append(avg_results)
            all_results.extend(sample_results)
        
        # Process noise config if specified
        if noise_config:
            logger.info("=" * 80)
            logger.info(f"🔬 NOISE RESILIENCE TESTING: {noise_config}")
            logger.info("=" * 80)
            
            # Load noise config
            config = load_config(noise_config)
            
            # Create model manager for efficient batch processing
            logger.info("🚀 Initializing models for noise testing (one-time load)...")
            model_manager = ModelManager(config)
            
            # Get BER levels for noise testing
            from benchmarks.noise_injection import get_ber_levels
            ber_levels = [(0.0, "No noise")] + get_ber_levels()
            logger.info(f"Testing {len(ber_levels)} noise levels per sample")
            
            # Process each LibriSpeech sample with noise
            sample_results = []
            
            for idx, sample_info in enumerate(librispeech_samples, 1):
                speaker_id = sample_info['speaker_id']
                chapter_id = sample_info['chapter_id']
                
                logger.info("=" * 60)
                logger.info(f"📖 Sample {idx}/{len(librispeech_samples)}: Speaker {speaker_id}, Chapter {chapter_id}")
                logger.info("=" * 60)
                
                # Merge FLAC files into single audio
                logger.info(f"Merging {len(sample_info['flac_files'])} FLAC files...")
                audio, sample_rate = merge_flac_files(sample_info['flac_files'])
                logger.info(f"Merged audio: {len(audio)/sample_rate:.1f}s @ {sample_rate} Hz")
                
                # Use first FLAC file as reference for voice cloning
                reference_file = str(sample_info['flac_files'][0])
                
                # Run benchmark for each noise level
                for ber, ber_desc in ber_levels:
                    if ber > 0:
                        logger.info(f"💥 Noise: {ber_desc} (BER = {ber:.2e})")
                    
                    # Benchmark our method with model_manager
                    our_results = benchmark_our_method(
                        audio, sample_rate, config, reference_file, 
                        bit_error_rate=ber, model_manager=model_manager
                    )
                    
                    # Add metadata
                    our_results['config'] = noise_config
                    our_results['config_data'] = config
                    our_results['noise_level'] = ber_desc
                    our_results['librispeech_speaker'] = speaker_id
                    our_results['librispeech_chapter'] = chapter_id
                    our_results['sample_index'] = idx
                    
                    # Calculate metrics if needed
                    if not args.skip_metrics and our_results.get('reconstruction_success'):
                        metrics = calculate_metrics(
                            audio,
                            our_results['reconstructed_audio'],
                            sample_rate,
                            our_results.get('original_text')
                        )
                        our_results.update(metrics)
                    
                    sample_results.append(our_results)
            
            # Calculate averaged statistics for noise config
            logger.info("=" * 80)
            logger.info(f"📊 AVERAGE STATISTICS for {noise_config} (with noise)")
            logger.info("=" * 80)
            
            # Group by noise level
            for ber, ber_desc in ber_levels:
                noise_level_results = [r for r in sample_results if r.get('noise_level') == ber_desc]
                
                if not noise_level_results:
                    continue
                
                # Calculate averages for numeric metrics
                avg_results = {'method': 'ours', 'config': noise_config, 'noise_level': ber_desc, 'is_average': True}
                numeric_keys = ['bitrate_bps', 'text_bps', 'prosody_bps', 'timbre_bps', 
                               'wer', 'speaker_similarity', 'pesq', 'stoi',
                               'nisqa_mos', 'nisqa_noisiness', 'nisqa_coloration', 
                               'nisqa_discontinuity', 'nisqa_loudness', 'duration',
                               'rtf', 'latency_total', 'latency_sender', 'latency_receiver']
                
                for key in numeric_keys:
                    values = [r[key] for r in noise_level_results if key in r and r[key] is not None]
                    if values:
                        avg_results[key] = np.mean(values)
                        avg_results[f'{key}_std'] = np.std(values)
                
                avg_results['num_samples'] = len(noise_level_results)
                avg_results['config_data'] = config
                
                # Log averaged results
                logger.info(f"\nAverage for {ber_desc} ({len(noise_level_results)} samples):")
                logger.info(f"  Bitrate: {avg_results.get('bitrate_bps', 0):.1f} ± {avg_results.get('bitrate_bps_std', 0):.1f} bps")
                if 'nisqa_mos' in avg_results:
                    logger.info(f"  NISQA MOS: {avg_results.get('nisqa_mos', 0):.3f} ± {avg_results.get('nisqa_mos_std', 0):.3f}")
                if 'wer' in avg_results:
                    logger.info(f"  WER: {avg_results.get('wer', 0):.3f} ± {avg_results.get('wer_std', 0):.3f}")
                
                sample_results.append(avg_results)
            
            # Extend main results with noise testing results
            all_results.extend(sample_results)
        
        # Save results
        if args.output:
            # Remove numpy arrays before JSON serialization
            for result in all_results:
                if 'reconstructed_audio' in result:
                    del result['reconstructed_audio']
            
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(all_results, f, indent=2, cls=NumpyEncoder)
            
            logger.info(f"✅ Results saved to: {output_path}")
        
        # Generate plots if in plot mode
        if args.plot and len(all_results) > 0:
            logger.info("=" * 80)
            logger.info("📊 GENERATING BEAUTIFUL PLOTS (LIBRISPEECH MODE)")
            logger.info("=" * 80)
            
            # Auto-save results in plot mode if not already specified
            plot_output_path = None
            if args.output:
                plot_output_path = output_path
            else:
                default_output = Path("librispeech_results.json")
                logger.info(f"Auto-saving plot results to: {default_output}")
                
                # Remove numpy arrays before JSON serialization
                results_copy = []
                for result in all_results:
                    result_copy = result.copy()
                    if 'reconstructed_audio' in result_copy:
                        del result_copy['reconstructed_audio']
                    results_copy.append(result_copy)
                
                with open(default_output, 'w') as f:
                    json.dump(results_copy, f, indent=2, cls=NumpyEncoder)
                
                plot_output_path = default_output
            
            try:
                from benchmarks.plot_results import generate_all_plots
                
                # Determine output directory
                plots_dir = plot_output_path.parent / "plots"
                
                # Filter to only use averaged results for plotting
                avg_results_only = [r for r in all_results if r.get('is_average')]
                if not avg_results_only:
                    logger.warning("No averaged results found for plotting. Using all results.")
                    avg_results_only = all_results
                
                generate_all_plots(avg_results_only, plots_dir)
                
                logger.info("=" * 80)
                logger.info("✅ PLOT GENERATION COMPLETE!")
                logger.info(f"📁 View your beautiful plots in: {plots_dir.absolute()}")
                logger.info("=" * 80)
                
            except ImportError as e:
                logger.warning(f"⚠️  Could not generate plots: {e}")
                logger.info("Install plotting dependencies: uv pip install matplotlib seaborn")
            except Exception as e:
                logger.error(f"❌ Error generating plots: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info("=" * 80)
        logger.info("✅ LIBRISPEECH BATCH BENCHMARK COMPLETE!")
        logger.info("=" * 80)
        
        return
    
    # Normal single audio processing mode
    # Determine which configs to run
    if args.config:
        configs_to_run = [args.config]
    else:
        # Find all YAML files in configs/ directory
        # Determine which directory to use
        if args.plot:
            configs_dir = Path("plot_configs")
        else:
            configs_dir = Path("configs")
        
        config_files = list(configs_dir.glob("*_mode.yaml"))
        configs_to_run = [f.stem for f in config_files]  # Get filename without .yaml
        
        # Filter based on --plot flag
        if args.plot:
            # Running all configs from plot_configs/
            logger.info(f"Plot mode enabled. Running plot configs from plot_configs/: {configs_to_run}")
        else:
            # Exclude plot_*_mode configs from configs/
            configs_to_run = [c for c in configs_to_run if not c.startswith("plot_")]
            logger.info(f"No config specified. Running all configs (excluding plot configs): {configs_to_run}")
    
    # Load audio once
    logger.info(f"Loading audio: {args.audio}")
    audio, sample_rate = load_audio(args.audio)
    logger.info(f"Audio: {len(audio)/sample_rate:.1f}s @ {sample_rate} Hz")
    
    reference_file = args.reference or args.audio
    if args.reference:
        logger.info(f"Using custom reference audio for voice cloning: {reference_file}")
    else:
        logger.info(f"Using input audio as TTS reference: {reference_file}")
    
    # Results storage
    all_results = []
    
    # Skip our method if baseline-only mode
    if not args.baseline_only:
        # Run benchmark for each config
        for config_name in configs_to_run:
            logger.info("=" * 80)
            logger.info(f"RUNNING CONFIG: {config_name}")
            logger.info("=" * 80)
            
            # Load config - use plot_configs/ directory if in plot mode
            if args.plot:
                config_path = Path("plot_configs") / f"{config_name}.yaml"
                config = load_config(config_path)
            else:
                config = load_config(config_name)
            logger.info(f"Loaded config: {config_name}")
            
            # Determine which BER levels to test
            if noise_config:
                from benchmarks.noise_injection import get_ber_levels
                ber_levels = [(0.0, "No noise")] + get_ber_levels()
                logger.info(f"Testing {len(ber_levels)} noise levels")
            else:
                ber_levels = [(0.0, "No noise")]
            
            # Run benchmark for each noise level
            for ber, ber_desc in ber_levels:
                if ber > 0:
                    logger.info("=" * 60)
                    logger.info(f"💥 NOISE LEVEL: {ber_desc} (BER = {ber:.2e})")
                    logger.info("=" * 60)
                
                # 1. Benchmark our method with specified BER
                our_results = benchmark_our_method(audio, sample_rate, config, reference_file, bit_error_rate=ber)
                
                # Add config name and data to results for clarity and plotting
                our_results['config'] = config_name
                our_results['config_data'] = config
                our_results['noise_level'] = ber_desc
                
                # Save reconstructed audio
                if our_results.get('reconstruction_success') and our_results.get('reconstructed_audio') is not None:
                    method_suffix = f"ours_{config_name}"
                    if ber > 0:
                        method_suffix += f"_ber{ber:.0e}".replace("-", "")
                    
                    saved_path = save_reconstructed_audio(
                        our_results['reconstructed_audio'],
                        sample_rate,
                        args.audio,
                        method_suffix
                    )
                    our_results['saved_audio_path'] = str(saved_path)
                
                if not args.skip_metrics and our_results.get('reconstruction_success'):
                    metrics = calculate_metrics(
                        audio,
                        our_results['reconstructed_audio'],
                        sample_rate,
                        our_results.get('original_text')
                    )
                    our_results.update(metrics)
                
                all_results.append(our_results)
    
    # 2. Benchmark baselines (only once, using first config's bandwidth)
    # Determine bandwidth to use for baselines
    if args.baseline_only or configs_to_run:
        # Load first config - use plot_configs/ directory if in plot mode
        if args.plot:
            first_config_path = Path("plot_configs") / f"{configs_to_run[0]}.yaml"
            first_config = load_config(first_config_path)
        else:
            first_config = load_config(configs_to_run[0])
        
        for baseline in args.baselines:
            logger.info("=" * 80)
            logger.info(f"RUNNING BASELINE: {baseline.upper()}")
            logger.info("=" * 80)
            
            baseline_results = None
            
            if baseline == 'opus':
                bandwidth = first_config.network.max_bandwidth_bps
                cache_path = get_baseline_cache_path(args.audio, 'opus', bandwidth)
                
                # Try to load from cache
                if not args.skip_baseline_cache:
                    baseline_results = load_baseline_cache(cache_path)
                
                # If not cached or cache skipped, run benchmark
                if baseline_results is None:
                    baseline_results = benchmark_baseline(
                        audio, sample_rate, 'opus',
                        bitrate=bandwidth
                    )
                    
                    # Save reconstructed audio
                    if baseline_results.get('reconstruction_success') and baseline_results.get('reconstructed_audio') is not None:
                        saved_path = save_reconstructed_audio(
                            baseline_results['reconstructed_audio'],
                            sample_rate,
                            args.audio,
                            'opus'
                        )
                        baseline_results['saved_audio_path'] = str(saved_path)
                    
                    if not args.skip_metrics and baseline_results.get('reconstruction_success'):
                        metrics = calculate_metrics(
                            audio,
                            baseline_results['reconstructed_audio'],
                            sample_rate
                        )
                        baseline_results.update(metrics)
                    
                    # Save to cache (without reconstructed_audio)
                    save_baseline_cache(cache_path, baseline_results)
                    
            elif baseline == 'encodec':
                # Convert bps to kbps
                bandwidth_kbps = first_config.network.max_bandwidth_bps / 1000
                # Round to closest Encodec bandwidth
                valid_bandwidths = [1.5, 3, 6, 12, 24]
                bandwidth_kbps = min(valid_bandwidths, key=lambda x: abs(x - bandwidth_kbps))
                
                cache_path = get_baseline_cache_path(args.audio, 'encodec', bandwidth_kbps)
                
                # Try to load from cache
                if not args.skip_baseline_cache:
                    baseline_results = load_baseline_cache(cache_path)
                
                # If not cached or cache skipped, run benchmark
                if baseline_results is None:
                    baseline_results = benchmark_baseline(
                        audio, sample_rate, 'encodec',
                        bandwidth=bandwidth_kbps
                    )
                    
                    # Save reconstructed audio
                    if baseline_results.get('reconstruction_success') and baseline_results.get('reconstructed_audio') is not None:
                        saved_path = save_reconstructed_audio(
                            baseline_results['reconstructed_audio'],
                            sample_rate,
                            args.audio,
                            'encodec'
                        )
                        baseline_results['saved_audio_path'] = str(saved_path)
                    
                    if not args.skip_metrics and baseline_results.get('reconstruction_success'):
                        metrics = calculate_metrics(
                            audio,
                            baseline_results['reconstructed_audio'],
                            sample_rate
                        )
                        baseline_results.update(metrics)
                    
                    # Save to cache (without reconstructed_audio)
                    save_baseline_cache(cache_path, baseline_results)
            else:
                logger.warning(f"Unknown baseline: {baseline}")
                continue
            
            if baseline_results:
                all_results.append(baseline_results)
    
    # 3. Print results
    print_results_table(all_results)
    
    # 4. Save results
    if args.output:
        # Remove numpy arrays before JSON serialization
        for result in all_results:
            if 'reconstructed_audio' in result:
                del result['reconstructed_audio']
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Results saved to: {output_path}")
    
    # 5. Generate plots if in plot mode
    if args.plot and len(all_results) > 0:
        logger.info("=" * 80)
        logger.info("📊 GENERATING BEAUTIFUL PLOTS")
        logger.info("=" * 80)
        
        # Auto-save results in plot mode if not already specified
        plot_output_path = None
        if args.output:
            plot_output_path = output_path
        else:
            default_output = Path("results.json")
            logger.info(f"Auto-saving plot results to: {default_output}")
            
            # Remove numpy arrays before JSON serialization
            results_copy = []
            for result in all_results:
                result_copy = result.copy()
                if 'reconstructed_audio' in result_copy:
                    del result_copy['reconstructed_audio']
                results_copy.append(result_copy)
            
            with open(default_output, 'w') as f:
                json.dump(results_copy, f, indent=2, cls=NumpyEncoder)
            
            plot_output_path = default_output
        
        try:
            from benchmarks.plot_results import generate_all_plots
            
            # Determine output directory
            plots_dir = plot_output_path.parent / "plots"
            
            generate_all_plots(all_results, plots_dir)
            
            logger.info("=" * 80)
            logger.info("✅ PLOT GENERATION COMPLETE!")
            logger.info(f"📁 View your beautiful plots in: {plots_dir.absolute()}")
            logger.info("=" * 80)
            
        except ImportError as e:
            logger.warning(f"⚠️  Could not generate plots: {e}")
            logger.info("Install plotting dependencies: uv pip install matplotlib seaborn")
        except Exception as e:
            logger.error(f"❌ Error generating plots: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
