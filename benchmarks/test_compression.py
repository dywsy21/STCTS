"""Quick test for compression pipeline only (no TTS reconstruction)."""

import sys
from pathlib import Path
import soundfile as sf
import librosa

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.stt.transcriber import WhisperTranscriber
from src.compression.text_compressor import TextCompressor
from src.prosody.extractor import ProsodyExtractor
from src.compression.prosody_compressor import ProsodyCompressor
from src.timbre.extractor import TimbreExtractor
from src.compression.timbre_compressor import TimbreCompressor


def test_compression_pipeline(audio_path: str, config_name: str):
    """Test the compression pipeline without TTS."""
    print(f"\n{'='*60}")
    print(f"Testing compression with {config_name}")
    print(f"{'='*60}\n")
    
    # 1. Load config
    config_path = Path("configs") / f"{config_name}.yaml"
    config = load_config(str(config_path))
    print(f"✅ Loaded config: {config_path}")
    
    # 2. Load audio
    audio, sr = sf.read(audio_path)
    duration = len(audio) / sr
    print(f"✅ Loaded audio: {duration:.1f}s @ {sr} Hz")
    
    # 3. STT
    print("\n1. Transcribing...")
    transcriber = WhisperTranscriber()
    text = transcriber.transcribe(audio, sr)
    print(f"   Text: \"{text[:80]}...\"")
    print(f"   Length: {len(text)} chars")
    
    # 4. Compress text
    print("\n2. Compressing text...")
    text_compressor = TextCompressor(algorithm=config.compression.text_algorithm)
    text_compressed = text_compressor.compress(text)
    text_bytes = len(text_compressed)
    text_bps = (text_bytes * 8) / duration
    print(f"   Size: {text_bytes} bytes")
    print(f"   Bitrate: {text_bps:.1f} bps")
    
    # 5. Extract and compress prosody
    print("\n3. Extracting and compressing prosody...")
    prosody_extractor = ProsodyExtractor(
        update_rate_hz=config.prosody.update_rate_hz,
        emotion_rate_hz=config.prosody.emotion_rate_hz,
        device="cpu"
    )
    prosody = prosody_extractor.extract(audio, sr)
    
    prosody_compressor = ProsodyCompressor(
        pitch_bits=config.compression.pitch_bits,
        energy_bits=config.compression.energy_bits,
        rate_bits=config.compression.rate_bits,
        include_emotion=config.prosody.include_emotion
    )
    
    # Simulate streaming: compress in chunks
    chunk_duration = 1.0 / config.prosody.update_rate_hz
    num_chunks = int(duration * config.prosody.update_rate_hz)
    
    total_prosody_bytes = 0
    for i in range(num_chunks):
        start_idx = i
        end_idx = min(i + 1, len(prosody['pitch']))
        
        chunk_prosody = {
            'pitch': prosody['pitch'][start_idx:end_idx],
            'energy': prosody['energy'][start_idx:end_idx],
            'rate': prosody['rate'][start_idx:end_idx],
        }
        
        if config.prosody.include_emotion:
            chunk_prosody['emotion'] = prosody['emotion'][start_idx:end_idx]
        
        compressed = prosody_compressor.compress(chunk_prosody)
        total_prosody_bytes += len(compressed)
    
    prosody_bps = (total_prosody_bytes * 8) / duration
    print(f"   Size: {total_prosody_bytes} bytes ({num_chunks} chunks)")
    print(f"   Bitrate: {prosody_bps:.1f} bps")
    
    # 6. Extract and compress timbre
    print("\n4. Extracting and compressing timbre...")
    timbre_extractor = TimbreExtractor(device="cpu")
    timbre = timbre_extractor.extract(audio, sr)
    
    timbre_compressor = TimbreCompressor()
    timbre_compressed = timbre_compressor.compress(timbre)
    timbre_bytes = len(timbre_compressed)
    timbre_bps = (timbre_bytes * 8) / duration
    print(f"   Size: {timbre_bytes} bytes")
    print(f"   Bitrate: {timbre_bps:.1f} bps")
    
    # 7. Total
    total_bytes = text_bytes + total_prosody_bytes + timbre_bytes
    total_bps = (total_bytes * 8) / duration
    
    print(f"\n{'='*60}")
    print(f"COMPRESSION SUMMARY")
    print(f"{'='*60}")
    print(f"Text:    {text_bytes:6d} bytes ({text_bps:6.1f} bps)")
    print(f"Prosody: {total_prosody_bytes:6d} bytes ({prosody_bps:6.1f} bps)")
    print(f"Timbre:  {timbre_bytes:6d} bytes ({timbre_bps:6.1f} bps)")
    print(f"{'-'*60}")
    print(f"Total:   {total_bytes:6d} bytes ({total_bps:6.1f} bps)")
    print(f"{'='*60}")
    
    if total_bps < 650:
        print(f"✅ PASS: Bitrate < 650 bps")
    else:
        print(f"❌ FAIL: Bitrate >= 650 bps")
    
    return total_bps


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default="test_audios/tpo53-1.wav")
    parser.add_argument("--config", default="minimal_mode")
    args = parser.parse_args()
    
    test_compression_pipeline(args.audio, args.config)
