# Benchmarking Guide

Comprehensive benchmarking system to evaluate STT-Compress-TTS against baseline codecs.

## Overview

The benchmarking system evaluates:
- **Bitrate**: Must be < 650 bps (strict requirement)
- **Intelligibility**: WER (Word Error Rate)
- **Voice Quality**: Speaker Similarity, PESQ, STOI, UTMOS
- **Comparison**: Against Opus, Encodec baselines

## Quick Start

```bash
# Run full benchmark with default config
uv run python -m benchmarks.run_all \
    --config minimal_mode \
    --audio test_audios/tpo53-1.wav

# Compare against specific baselines
uv run python -m benchmarks.run_all \
    --config balanced_mode \
    --audio test_audios/tpo53-1.wav \
    --baselines opus encodec

# Save results to JSON
uv run python -m benchmarks.run_all \
    --config minimal_mode \
    --audio test_audios/tpo53-1.wav \
    --output results/benchmark_minimal.json
```

## Metrics Explained

### 1. Bitrate (bps)

**Target: < 650 bps (CRITICAL)**

Measures data throughput required for transmission:
- Text: Compressed transcription
- Prosody: Pitch/energy/rate features (sent at update_rate_hz)
- Timbre: Speaker embedding (sent once per speaker/minute)

**Our approach components:**
- Text: ~50-150 bps (bursty, per sentence)
- Prosody: ~200-500 bps (continuous streaming)
- Timbre: ~50 bps (rare updates)

**Controlling bitrate:**
- Reduce `prosody.update_rate_hz` (e.g., 2.5 → 1.0 Hz)
- Lower quantization bits (e.g., 6 → 4 bits)
- Use faster text compression (brotli → lz4)
- Increase timbre update interval (60s → 120s)

### 2. WER (Word Error Rate)

**Range: 0.0 to ∞ (lower is better)**
**Target: < 0.1 (10% error rate)**

Measures speech intelligibility:
```
WER = (Substitutions + Deletions + Insertions) / Total Words
```

- **0.0-0.05**: Excellent intelligibility
- **0.05-0.10**: Good intelligibility
- **0.10-0.20**: Acceptable for most applications
- **> 0.20**: Poor intelligibility

Calculated by:
1. Transcribe original audio with Whisper
2. Transcribe reconstructed audio with Whisper
3. Compare transcriptions using edit distance

### 3. Speaker Similarity (SpkrSim)

**Range: -1.0 to 1.0 (higher is better)**
**Target: > 0.85 (same speaker threshold)**

Measures how well speaker identity is preserved:
- Uses cosine similarity between speaker embeddings
- **> 0.85**: Recognized as same speaker
- **0.70-0.85**: Similar voice characteristics
- **< 0.70**: Different speaker

### 4. PESQ (Perceptual Evaluation of Speech Quality)

**Range: -0.5 to 4.5 (higher is better)**
**Target: > 2.5**

ITU-T standard for speech quality:
- **3.5-4.5**: Excellent quality
- **2.5-3.5**: Good quality
- **1.5-2.5**: Fair quality
- **< 1.5**: Poor quality

Intrusive metric (requires reference audio).

### 5. STOI (Short-Time Objective Intelligibility)

**Range: 0.0 to 1.0 (higher is better)**
**Target: > 0.85**

Measures speech intelligibility:
- **> 0.90**: Excellent intelligibility
- **0.85-0.90**: Good intelligibility
- **0.70-0.85**: Fair intelligibility
- **< 0.70**: Poor intelligibility

Correlates well with subjective listening tests.

### 6. UTMOS (Universal TTS MOS)

**Range: 1.0 to 5.0 (higher is better)**
**Target: > 3.5**

Predicts Mean Opinion Score (MOS):
- **4.0-5.0**: Excellent naturalness
- **3.0-4.0**: Good naturalness
- **2.0-3.0**: Fair naturalness
- **< 2.0**: Poor naturalness

Non-intrusive (no reference needed).

## Baselines

### Opus

Traditional audio codec optimized for speech:
- Variable bitrate: 6-510 kbps
- Low latency
- Good quality at low bitrates
- Widely used in VoIP

**Installation:**
```bash
# Ubuntu/Debian
sudo apt install opus-tools

# macOS
brew install opus-tools

# Windows
# Download from: https://opus-codec.org/downloads/
```

### Encodec

Meta's neural audio codec:
- Variable bandwidth: 1.5-24 kbps
- State-of-the-art neural compression
- Good for music and speech
- Higher computational cost

**Installation:**
```bash
pip install encodec
```

### Vevo (TODO)

Voice conversion-based codec (to be implemented).

## Installation

### Core Dependencies

```bash
# Already included in requirements.txt
pip install pesq pystoi encodec scikit-learn
```

### Optional: UTMOS

```bash
# UTMOS uses HuggingFace transformers (already installed)
# Model will auto-download on first use
```

### Optional: NISQA

```bash
# Clone and install NISQA
git clone https://github.com/gabrielmittag/NISQA
cd NISQA
pip install -e .
```

### System Dependencies

```bash
# Opus tools (Ubuntu/Debian)
sudo apt install opus-tools

# Opus tools (macOS)
brew install opus-tools
```

## Advanced Usage

### Custom Reference Audio

Use different audio for TTS voice cloning:

```bash
uv run python -m benchmarks.run_all \
    --config minimal_mode \
    --audio test_audios/tpo53-1.wav \
    --reference test_audios/tpo53-4.wav
```

### Skip Metrics (Faster)

For quick bitrate-only testing:

```bash
uv run python -m benchmarks.run_all \
    --config minimal_mode \
    --audio test_audios/tpo53-1.wav \
    --skip-metrics
```

### Batch Testing

Test multiple configs:

```bash
for config in minimal_mode balanced_mode high_quality_mode; do
    uv run python -m benchmarks.run_all \
        --config $config \
        --audio test_audios/tpo53-1.wav \
        --output results/benchmark_$config.json
done
```

## Interpreting Results

### Example Output

```
📊 BITRATE COMPARISON
--------------------------------------------------------------------------------
Method          Bitrate (bps)   Within Target  
--------------------------------------------------------------------------------
ours            425.3           ✅ YES          
opus            6000.0          ❌ NO           
encodec         1500.0          ❌ NO           

🎯 QUALITY METRICS
--------------------------------------------------------------------------------
Method          WER        SpkrSim    PESQ       STOI       UTMOS     
--------------------------------------------------------------------------------
ours            0.045      0.892      3.2        0.88       3.8       
opus            0.012      0.945      4.1        0.95       4.3       
encodec         0.034      0.912      3.6        0.91       4.0       
```

### Analysis

**Our method (425.3 bps):**
- ✅ **Meets bitrate target** (< 650 bps)
- ✅ **Good WER** (4.5% error rate)
- ✅ **Good speaker similarity** (0.892 > 0.85)
- ⚠️ **Moderate PESQ** (3.2 - acceptable quality)
- ✅ **Good STOI** (0.88 - good intelligibility)
- ✅ **Good UTMOS** (3.8 - natural sounding)

**Opus (6000 bps):**
- ❌ **10x over bitrate target**
- ✅ Excellent quality metrics across the board
- Traditional codec baseline

**Encodec (1500 bps):**
- ❌ **2.4x over bitrate target**
- ✅ Very good quality metrics
- Neural codec baseline

### Trade-offs

Our ultra-low bitrate approach sacrifices some quality for extreme compression:

**Advantages:**
- 10-15x lower bitrate than Opus
- 3-5x lower bitrate than Encodec
- Good intelligibility (WER, STOI)
- Preserves speaker identity
- Acceptable perceptual quality

**Disadvantages:**
- Lower PESQ than traditional codecs
- Prosody may be less accurate
- Background noise not preserved
- Higher computational cost (TTS synthesis)

## Troubleshooting

### PESQ Installation Issues

```bash
# PESQ requires compilation
pip install --upgrade pip setuptools wheel
pip install pesq
```

### Opus Not Found

```bash
# Ensure opus-tools is installed and in PATH
which opusenc  # Should show path to binary
```

### CUDA Out of Memory

```bash
# Use CPU for TTS
export CUDA_VISIBLE_DEVICES=""

# Or edit benchmarks/run_all.py to use device="cpu"
```

### Encodec Import Error

```bash
# Reinstall encodec
pip uninstall encodec
pip install encodec
```

## Future Work

- [ ] Implement Vevo baseline
- [ ] Add NISQA integration
- [ ] Add speaker recognition accuracy (SCA)
- [ ] Multi-speaker test sets
- [ ] Confidence intervals for metrics
- [ ] Perceptual A/B listening tests
- [ ] Real-time streaming simulation

## Citation

If you use this benchmarking system, please cite:

```bibtex
@software{stt_compress_tts_bench,
  title = {STT-Compress-TTS Benchmarking Suite},
  author = {Your Team},
  year = {2025},
  url = {https://github.com/yourusername/stt-compress-tts}
}
```
