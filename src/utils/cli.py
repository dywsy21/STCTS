"""Command-line interface for STT-Compress-TTS."""

import asyncio
import argparse
import sys
from typing import Optional

from src.utils.logger import get_logger
from src.config import Settings

logger = get_logger(__name__)


async def run_sender(config_path: str, quality_mode: str):
    """Run sender pipeline.
    
    Args:
        config_path: Path to config file
        quality_mode: Quality mode (minimal/balanced/high)
    """
    from src.pipeline import SenderPipeline
    from src.network import PacketPriorityQueue
    
    logger.info(f"Starting sender in {quality_mode} mode")
    
    # Load config
    settings = Settings()
    
    # Create packet queue
    packet_queue = PacketPriorityQueue()
    
    # Create sender pipeline
    sender = SenderPipeline(
        config=settings.dict(),
        packet_queue=packet_queue
    )
    
    try:
        await sender.start()
        logger.info("Sender started. Press Ctrl+C to stop.")
        
        # Run until interrupted
        while True:
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Stopping sender...")
    finally:
        await sender.stop()


async def run_receiver(config_path: str):
    """Run receiver pipeline.
    
    Args:
        config_path: Path to config file
    """
    from src.pipeline import ReceiverPipeline
    
    logger.info("Starting receiver")
    
    # Load config
    settings = Settings()
    
    # Create receiver pipeline
    receiver = ReceiverPipeline(config=settings.dict())
    
    try:
        await receiver.start()
        logger.info("Receiver started. Press Ctrl+C to stop.")
        
        # Run until interrupted
        while True:
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Stopping receiver...")
    finally:
        await receiver.stop()


async def run_signaling_server(host: str, port: int):
    """Run signaling server.
    
    Args:
        host: Server host
        port: Server port
    """
    from src.signaling import SignalingServer
    
    logger.info(f"Starting signaling server on {host}:{port}")
    
    server = SignalingServer(host=host, port=port)
    
    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Stopping signaling server...")


async def test_compression(input_file: str, reference_file: Optional[str] = None, config_path: Optional[str] = None):
    """Test compression on audio file.
    
    Args:
        input_file: Path to input audio file
        reference_file: Path to reference audio file for voice cloning (defaults to input_file)
        config_path: Path to config YAML file (can be relative name or full path)
    """
    from src.stt import FasterWhisperSTT
    from src.compression.text import TextCompressor
    from src.compression.prosody import ProsodyCompressor
    from src.compression.timbre import TimbreCompressor
    from src.prosody import ProsodyExtractor
    from src.speaker import SpeakerEmbedding
    from src.utils.config import load_config
    import soundfile as sf
    import librosa
    import numpy as np
    from pathlib import Path
    
    # Load configuration
    if config_path:
        # Check if it's just a name (e.g., "minimal_mode") or full path
        config_path_obj = Path(config_path)
        if not config_path_obj.exists():
            # Try in configs/ folder
            if not config_path.endswith('.yaml'):
                config_path = f"{config_path}.yaml"
            config_path_obj = Path("configs") / config_path
        
        if config_path_obj.exists():
            config = load_config(config_path_obj)
            logger.info(f"📋 Using config: {config_path_obj}")
        else:
            logger.warning(f"Config file not found: {config_path}, using default config")
            config = load_config()
    else:
        config = load_config()
        logger.info("📋 Using default config (balanced_mode)")
    
    # Display key compression settings
    logger.info("=" * 60)
    logger.info("COMPRESSION SETTINGS:")
    logger.info(f"  Text: {config.compression.text_algorithm} (level {config.compression.text_level})")
    logger.info(f"  Prosody: pitch={config.compression.prosody_quantization_pitch_bits}bits, "
                f"energy={config.compression.prosody_quantization_energy_bits}bits, "
                f"rate={config.compression.prosody_quantization_rate_bits}bits")
    logger.info(f"  Timbre: {config.compression.timbre_algorithm}")
    logger.info(f"  Max Bandwidth: {config.network.max_bandwidth_bps} bps")
    logger.info("=" * 60)
    
    logger.info(f"Testing compression on {input_file}")
    
    # Set reference file for TTS (default to input_file if not provided)
    if reference_file is None:
        reference_file = input_file
        logger.info(f"Using input file as TTS reference: {reference_file}")
    else:
        logger.info(f"Using custom TTS reference: {reference_file}")
    
    # Load audio
    audio, sr = sf.read(input_file)
    logger.info(f"Audio: {len(audio)} samples, {sr} Hz")
    
    # Resample to 16kHz if needed (STT expects 16kHz)
    target_sr = 16000
    if sr != target_sr:
        logger.info(f"Resampling from {sr} Hz to {target_sr} Hz")
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
        logger.info(f"Resampled audio: {len(audio)} samples")
    
    # STT
    stt = FasterWhisperSTT()
    result = stt.transcribe(audio, sr)
    text = result.text
    logger.info(f"Transcribed text: {text}")
    
    # Check if text is empty
    if not text or len(text.strip()) == 0:
        logger.error("No text transcribed! Audio may be silent or contain no speech.")
        return
    
    # Compress text using config settings
    text_compressor = TextCompressor(
        algorithm=config.compression.text_algorithm,
        level=config.compression.text_level,
        preprocess=config.compression.text_preprocess
    )
    compressed_text = text_compressor.compress(text)
    text_ratio = len(compressed_text) / len(text.encode())
    logger.info(f"Text: {len(text)} chars -> {len(compressed_text)} bytes (ratio: {text_ratio:.2f})")
    
    # Extract and compress prosody IN CHUNKS (simulating real streaming)
    logger.info("=" * 60)
    logger.info("PROSODY COMPRESSION (Streaming Simulation)")
    logger.info(f"Update rate: {config.prosody.update_rate_hz} Hz (every {1.0/config.prosody.update_rate_hz:.2f}s)")
    logger.info("=" * 60)
    
    prosody_extractor = ProsodyExtractor(sample_rate=sr)
    
    # Compress prosody using config settings
    prosody_compressor = ProsodyCompressor(
        pitch_bits=config.compression.prosody_quantization_pitch_bits,
        energy_bits=config.compression.prosody_quantization_energy_bits,
        rate_bits=config.compression.prosody_quantization_rate_bits,
        use_delta_encoding=config.compression.prosody_delta_encoding
    )
    
    # Simulate streaming: split audio into chunks based on update_rate_hz
    update_rate = config.prosody.update_rate_hz
    chunk_duration = 1.0 / update_rate  # seconds per chunk
    chunk_samples = int(sr * chunk_duration)
    duration = len(audio) / sr
    num_chunks = int(duration * update_rate)
    
    prosody_packets = []
    total_prosody_bytes = 0
    
    logger.info(f"Splitting {duration:.1f}s audio into {num_chunks} chunks of {chunk_duration:.2f}s each")
    
    for i in range(num_chunks):
        start_idx = i * chunk_samples
        end_idx = min(start_idx + chunk_samples, len(audio))
        audio_chunk = audio[start_idx:end_idx]
        
        if len(audio_chunk) < sr * 0.1:  # Skip chunks < 0.1s
            continue
        
        # Extract prosody for this chunk
        prosody_chunk = prosody_extractor.extract_all(audio_chunk)
        
        # Compress this chunk
        compressed_pitch = prosody_compressor.compress_pitch(prosody_chunk.pitch) if prosody_chunk.pitch is not None else b""
        compressed_energy = prosody_compressor.compress_energy(prosody_chunk.energy) if prosody_chunk.energy is not None else b""
        
        chunk_bytes = len(compressed_pitch) + len(compressed_energy)
        prosody_packets.append(chunk_bytes)
        total_prosody_bytes += chunk_bytes
    
    avg_packet_size = total_prosody_bytes / len(prosody_packets) if prosody_packets else 0
    min_packet = min(prosody_packets) if prosody_packets else 0
    max_packet = max(prosody_packets) if prosody_packets else 0
    
    logger.info(f"Prosody packets: {len(prosody_packets)} packets, {total_prosody_bytes} bytes total")
    logger.info(f"  Average packet: {avg_packet_size:.1f} bytes")
    logger.info(f"  Min/Max packet: {min_packet}/{max_packet} bytes")
    logger.info(f"  Prosody bandwidth: {(total_prosody_bytes * 8) / duration:.1f} bps")

    
    # Extract and compress speaker embedding
    speaker_embedding = SpeakerEmbedding()
    embedding = speaker_embedding.extract(audio, sr)
    
    # Compress timbre using config settings
    timbre_compressor = TimbreCompressor(algorithm=config.compression.timbre_algorithm)
    compressed_timbre = timbre_compressor.compress(embedding)
    logger.info(f"Timbre: {len(compressed_timbre)} bytes")
    
    # Calculate total bitrate (streaming with chunked prosody)
    logger.info("=" * 60)
    duration = len(audio) / sr
    total_bytes = len(compressed_text) + total_prosody_bytes + len(compressed_timbre)
    bitrate = (total_bytes * 8) / duration
    logger.info(f"TOTAL COMPRESSED: {total_bytes} bytes for {duration:.2f}s")
    logger.info(f"  Text: {len(compressed_text)}B ({len(compressed_text)/total_bytes*100:.1f}%)")
    logger.info(f"  Prosody: {total_prosody_bytes}B in {len(prosody_packets)} packets ({total_prosody_bytes/total_bytes*100:.1f}%)")
    logger.info(f"  Timbre: {len(compressed_timbre)}B ({len(compressed_timbre)/total_bytes*100:.1f}%)")
    logger.info(f"Average bitrate: {bitrate:.2f} bps")
    logger.info("=" * 60)
    
    # Calculate STREAMING bandwidth (realistic usage)
    logger.info("=" * 60)
    logger.info("STREAMING BANDWIDTH ESTIMATION (Real-world usage)")
    logger.info("=" * 60)
    
    # Text bandwidth (per sentence, bursty)
    num_sentences = text.count('.') + text.count('!') + text.count('?')
    if num_sentences == 0:
        num_sentences = 1
    text_bps = (len(compressed_text) * 8) / duration
    logger.info(f"Text: {len(compressed_text)}B over {duration:.1f}s = {text_bps:.1f} bps (bursty, ~{num_sentences} packets)")
    
    # Prosody bandwidth (use actual measured from chunked compression)
    prosody_bps = (total_prosody_bytes * 8) / duration
    logger.info(f"Prosody: {total_prosody_bytes}B in {len(prosody_packets)} packets = {prosody_bps:.1f} bps (measured)")
    logger.info(f"  Avg packet: {avg_packet_size:.1f}B × {update_rate} Hz = {avg_packet_size * 8 * update_rate:.1f} bps theoretical")
    
    # Timbre bandwidth (sent once per minute or on speaker change)
    timbre_update_interval = 60  # seconds
    timbre_bps = (len(compressed_timbre) * 8) / timbre_update_interval
    logger.info(f"Timbre: {len(compressed_timbre)}B / {timbre_update_interval}s = {timbre_bps:.1f} bps (rare)")
    
    # Emotion bandwidth (if enabled)
    emotion_rate = config.prosody.emotion_rate_hz if hasattr(config.prosody, 'emotion_rate_hz') else 0
    emotion_bytes_per_packet = 20  # emotion label + confidence
    emotion_bps = emotion_bytes_per_packet * 8 * emotion_rate
    if emotion_rate > 0:
        logger.info(f"Emotion: ~{emotion_bytes_per_packet}B × {emotion_rate} Hz = {emotion_bps:.1f} bps (optional)")
    
    # Total streaming bandwidth
    total_streaming_bps = text_bps + prosody_bps + timbre_bps + emotion_bps
    target_bps = config.network.max_bandwidth_bps
    logger.info("=" * 60)
    logger.info(f"TOTAL STREAMING BANDWIDTH: {total_streaming_bps:.1f} bps")
    logger.info(f"TARGET BANDWIDTH: {target_bps} bps")
    
    if total_streaming_bps <= target_bps:
        logger.info(f"✅ Within target! (using {total_streaming_bps/target_bps*100:.1f}% of bandwidth)")
    else:
        logger.info(f"⚠️  Over target by {total_streaming_bps - target_bps:.1f} bps ({total_streaming_bps/target_bps*100:.1f}% of target)")
    logger.info("=" * 60)
    
    # ========== RECEIVER SIMULATION ==========
    logger.info("=" * 60)
    logger.info("Simulating receiver end - decompression and TTS")
    logger.info("=" * 60)
    
    # Decompress text
    decompressed_text = text_compressor.decompress(compressed_text)
    logger.info(f"Decompressed text: {decompressed_text[:100]}..." if len(decompressed_text) > 100 else f"Decompressed text: {decompressed_text}")
    
    # Decompress speaker embedding
    decompressed_embedding = timbre_compressor.decompress(compressed_timbre, dim=192)
    logger.info(f"Decompressed speaker embedding: shape {decompressed_embedding.shape}")
    
    # Decompress prosody packets and reconstruct full contour
    logger.info(f"Decompressing {len(prosody_packets)} prosody packets...")
    
    decompressed_pitch_chunks = []
    decompressed_energy_chunks = []
    
    # For decompression, we need to store the compressed data properly
    # Let's reconstruct by re-compressing each chunk (we have the original chunks)
    for i in range(num_chunks):
        start_idx = i * chunk_samples
        end_idx = min(start_idx + chunk_samples, len(audio))
        audio_chunk = audio[start_idx:end_idx]
        
        if len(audio_chunk) < sr * 0.1:  # Skip chunks < 0.1s
            continue
        
        # Extract prosody for this chunk (we do this again to get original data)
        prosody_chunk = prosody_extractor.extract_all(audio_chunk)
        
        # Compress
        compressed_pitch = prosody_compressor.compress_pitch(prosody_chunk.pitch) if prosody_chunk.pitch is not None else b""
        compressed_energy = prosody_compressor.compress_energy(prosody_chunk.energy) if prosody_chunk.energy is not None else b""
        
        # Decompress (simulating receiver)
        if compressed_pitch:
            # Get original length for decompression
            original_pitch_len = len(prosody_chunk.pitch) if prosody_chunk.pitch is not None else 0
            if original_pitch_len > 0:
                decompressed_pitch = prosody_compressor.decompress_pitch(compressed_pitch, length=original_pitch_len)
                decompressed_pitch_chunks.append(decompressed_pitch)
        
        if compressed_energy:
            original_energy_len = len(prosody_chunk.energy) if prosody_chunk.energy is not None else 0
            if original_energy_len > 0:
                decompressed_energy = prosody_compressor.decompress_energy(compressed_energy, length=original_energy_len)
                decompressed_energy_chunks.append(decompressed_energy)
    
    # Concatenate all decompressed chunks
    if decompressed_pitch_chunks:
        full_decompressed_pitch = np.concatenate(decompressed_pitch_chunks)
        logger.info(f"Decompressed pitch contour: {len(full_decompressed_pitch)} frames")
    else:
        full_decompressed_pitch = None
        
    if decompressed_energy_chunks:
        full_decompressed_energy = np.concatenate(decompressed_energy_chunks)
        logger.info(f"Decompressed energy contour: {len(full_decompressed_energy)} frames")
    else:
        full_decompressed_energy = None
    
    # Package prosody for TTS (even though XTTS doesn't use it directly)
    reconstructed_prosody = {
        'pitch': full_decompressed_pitch,
        'energy': full_decompressed_energy,
        'speaking_rate': None  # Not implemented yet
    }
    logger.info(f"Reconstructed prosody: pitch={'✓' if full_decompressed_pitch is not None else '✗'}, energy={'✓' if full_decompressed_energy is not None else '✗'}")
    
    # Prepare output path
    input_path = Path(input_file)
    
    # Save decompressed prosody for analysis
    prosody_output_path = input_path.parent / f"{input_path.stem}_prosody_reconstructed.npz"
    np.savez(
        prosody_output_path,
        pitch=full_decompressed_pitch if full_decompressed_pitch is not None else np.array([]),
        energy=full_decompressed_energy if full_decompressed_energy is not None else np.array([])
    )
    logger.info(f"💾 Decompressed prosody saved to: {prosody_output_path}")
    
    # Synthesize speech with TTS
    try:
        from src.tts.xtts import XTTSTS
        
        logger.info("Loading TTS model (this may take a moment)...")
        tts = XTTSTS(device="cpu")  # Use CPU for compatibility
        
        # Synthesize using reference audio for speaker voice cloning
        # Note: XTTS-v2 doesn't use prosody directly, but we pass it for future models
        logger.info("Synthesizing speech...")
        logger.info("Note: XTTS-v2 uses speaker cloning, prosody is saved separately for analysis")
        tts_result = tts.synthesize(
            text=decompressed_text,
            speaker_embedding=decompressed_embedding,
            prosody=reconstructed_prosody,  # Passed but not used by XTTS-v2
            speaker_wav=str(reference_file)  # Use reference audio for voice cloning
        )
        
        # Save reconstructed audio
        output_path = input_path.parent / f"{input_path.stem}_reconstructed.wav"
        
        sf.write(output_path, tts_result.audio, tts_result.sample_rate)
        logger.info(f"✅ Reconstructed audio saved to: {output_path}")
        logger.info(f"   Duration: {tts_result.duration:.2f}s, Sample rate: {tts_result.sample_rate} Hz")
        
    except ImportError as e:
        logger.warning(f"TTS not available (missing dependency): {e}")
        logger.warning("Install with: uv pip install TTS")
        
        # Save text transcript as fallback
        text_output_path = input_path.parent / f"{input_path.stem}_reconstructed.txt"
        with open(text_output_path, 'w', encoding='utf-8') as f:
            f.write(decompressed_text)
        logger.info(f"💾 Decompressed text saved to: {text_output_path}")
        
    except Exception as e:
        logger.error(f"Error during TTS synthesis: {e}")
        logger.info("Skipping audio reconstruction (text decompression successful)")
        
        # Save text transcript as fallback
        text_output_path = input_path.parent / f"{input_path.stem}_reconstructed.txt"
        with open(text_output_path, 'w', encoding='utf-8') as f:
            f.write(decompressed_text)
        logger.info(f"💾 Decompressed text saved to: {text_output_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="STT-Compress-TTS CLI")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Sender command
    sender_parser = subparsers.add_parser("sender", help="Run sender pipeline")
    sender_parser.add_argument("--config", default=None, help="Path to config file")
    sender_parser.add_argument(
        "--quality",
        choices=["minimal", "balanced", "high"],
        default="balanced",
        help="Quality mode"
    )
    
    # Receiver command
    receiver_parser = subparsers.add_parser("receiver", help="Run receiver pipeline")
    receiver_parser.add_argument("--config", default=None, help="Path to config file")
    
    # Signaling server command
    signaling_parser = subparsers.add_parser("signaling", help="Run signaling server")
    signaling_parser.add_argument("--host", default="0.0.0.0", help="Server host")
    signaling_parser.add_argument("--port", type=int, default=8080, help="Server port")
    
    # Test compression command
    test_parser = subparsers.add_parser("test", help="Test compression on audio file")
    test_parser.add_argument("input_file", help="Path to input audio file")
    test_parser.add_argument(
        "--reference",
        "-r",
        default=None,
        help="Path to reference audio file for voice cloning (defaults to input_file)"
    )
    test_parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Config file path or name (e.g., 'minimal_mode', 'configs/balanced_mode.yaml')"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Run command
    try:
        if args.command == "sender":
            asyncio.run(run_sender(args.config, args.quality))
        
        elif args.command == "receiver":
            asyncio.run(run_receiver(args.config))
        
        elif args.command == "signaling":
            asyncio.run(run_signaling_server(args.host, args.port))
        
        elif args.command == "test":
            asyncio.run(test_compression(args.input_file, args.reference, args.config))
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
