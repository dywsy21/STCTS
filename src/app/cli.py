"""CLI application entry point."""

import sys
from pathlib import Path

import click

from src.utils import settings, setup_logging
from src.utils.logger import get_logger

# Setup logging
setup_logging(
    log_level=settings.log_level,
    log_file=settings.log_file,
)

logger = get_logger(__name__)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """STT-Compression-TTS Voice Communication System."""
    pass


@cli.command()
@click.option("--peer-id", required=True, help="Peer ID for this client")
@click.option("--signaling", default="ws://localhost:8765", help="Signaling server URL (e.g., ws://localhost:8765)")
@click.option("--connect-to", help="Peer ID to automatically connect to")
@click.option("--quality-mode", default="balanced", help="Quality mode (minimal, balanced, high)")
@click.option("--config", help="Path to config file (overrides --quality-mode)")
@click.option("--output-audio-only", is_flag=True, help="Receiver-only mode: only play audio, don't send")
@click.option("--quiet", is_flag=True, help="Reduce log output (only warnings and errors)")
def client(peer_id: str, signaling: str, connect_to: str, quality_mode: str, config: str, output_audio_only: bool, quiet: bool):
    """Start a voice call client."""
    import asyncio
    import logging
    from src.app.voice_client import VoiceClient
    
    # Adjust log level if quiet mode
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)
        click.echo("📢 Quiet mode: Only warnings and errors will be shown")
    
    mode_str = "receiver-only" if output_audio_only else "full-duplex"
    config_str = config if config else f"{quality_mode} mode"
    logger.info(f"Starting client: peer_id={peer_id}, signaling={signaling}, config={config_str}, {mode_str}")
    
    # Create and run voice client
    voice_client = VoiceClient(
        peer_id=peer_id,
        signaling_url=signaling,
        quality_mode=quality_mode,
        config_path=config,
        auto_connect_peer=connect_to,
        output_audio_only=output_audio_only
    )
    
    try:
        asyncio.run(voice_client.run_interactive())
    except KeyboardInterrupt:
        click.echo("\n\nShutting down...")
    finally:
        click.echo("Client stopped.")


@cli.command()
@click.option("--input-file", type=click.Path(exists=True), required=True, help="Input audio file")
@click.option("--output-file", type=click.Path(), required=True, help="Output audio file")
@click.option("--quality-mode", default="balanced", help="Quality mode")
def test_pipeline(input_file: str, output_file: str, quality_mode: str):
    """Test the STT-Compression-TTS pipeline with a file."""
    from src.utils.config import load_config
    from src.stt.faster_whisper import FasterWhisperSTT
    from src.compression.text import TextCompressor
    from src.tts.xtts import XTTSTS
    
    logger.info("Testing pipeline...")
    
    # Load config
    config = load_config(quality_mode=quality_mode)
    
    # Initialize models
    click.echo("Loading models...")
    stt = FasterWhisperSTT(model_size=config.stt.model_size)
    text_compressor = TextCompressor(
        algorithm=config.compression.text_algorithm,
        level=config.compression.text_level,
    )
    tts = XTTSTS()
    
    # Load audio
    click.echo(f"Loading audio from {input_file}...")
    import soundfile as sf
    audio, sample_rate = sf.read(input_file)
    
    # STT
    click.echo("Transcribing...")
    result = stt.transcribe(audio, sample_rate)
    click.echo(f"Text: {result.text}")
    
    # Compress
    click.echo("Compressing...")
    compressed = text_compressor.compress(result.text)
    click.echo(f"Compressed size: {len(compressed)} bytes")
    
    # Decompress
    decompressed_text = text_compressor.decompress(compressed)
    click.echo(f"Decompressed text: {decompressed_text}")
    
    # TTS
    click.echo("Synthesizing...")
    tts_result = tts.synthesize(decompressed_text)
    
    # Save
    click.echo(f"Saving audio to {output_file}...")
    sf.write(output_file, tts_result.audio, tts_result.sample_rate)
    
    click.echo("✓ Pipeline test complete!")


@cli.command()
@click.option("--duration", default=30, help="Collection duration in seconds")
@click.option("--output", type=click.Path(), required=True, help="Output profile path")
def collect_timbre(duration: int, output: str):
    """Collect timbre profile from microphone."""
    click.echo(f"🎤 Collecting timbre profile for {duration} seconds...")
    click.echo("Speak naturally into your microphone.")
    click.echo()
    
    # TODO: Implement timbre collection
    # - Capture audio from microphone
    # - Extract speaker embedding
    # - Save profile
    
    click.echo(f"Profile will be saved to: {output}")


@cli.command()
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--port", default=8765, help="Server port")
def server(host: str, port: int):
    """Start the signaling server."""
    import asyncio
    from src.signaling.server import SignalingServer
    
    logger.info(f"Starting signaling server on {host}:{port}")
    
    click.echo("=" * 60)
    click.echo("🌐 STT-Compress-TTS Signaling Server")
    click.echo(f"Listening on: ws://{host}:{port}")
    click.echo("=" * 60)
    click.echo()
    
    signaling_server = SignalingServer(host=host, port=port)
    
    try:
        asyncio.run(signaling_server.start())
    except KeyboardInterrupt:
        click.echo("\n\nShutting down server...")
    finally:
        click.echo("Server stopped.")


def main():
    """Main entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
