# STCTS: An Ultra-Low Bitrate Voice Communication System

Ultra-low bandwidth voice communication system achieving **70~80 bps** using Speech-to-Text, intelligent compression, and Text-to-Speech.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

## Demo

To showcase our STCTS system, here's an audio piece from Librispeech that is processed through Opus, Encodec and STCTS:

| Method | Bitrate | Audio |
| :--- | :--- | :--- |
| **Original** | 256 kbps | <audio controls src="demo/260-123286-0025.wav"></audio> |
| **EnCodec** | 6 kbps | <audio controls src="demo/260-123286-0025_encodec_20251120_232801.wav"></audio> |
| **Opus** | 1 kbps | <audio controls src="demo/260-123286-0025_opus_20251120_232755.wav"></audio> |
| **STCTS (Ours)** | 90.4 bps* | <audio controls src="demo/260-123286-0025_ours_high_quality_20251120_232746.wav"></audio> |

_*w/o timbre_

## Introduction

### Background

In some regions on the Earth and amidst ocean and the heavens, network bandwidth is extremely low and precious: an end-to-end low bandwidth dual-way voice call transference system that uses as few bandwidth resources as possible is needed to allow for convenient and cheap voice calls in those scenarios.

### Our Approach: STT-Compression-TTS

1. **STT** to convert voice into text while also extract prosody and timbre
2. **Compression** to minimize text and metadata (prosody, timbre)
3. **TTS** to synthesize voice on the receiving end 

Refer to the [paper](paper/STCTS--Generative%20Semantic%20Compression%20for%20Ultra-Low%20Bitrate%20Speech%20via%20Explicit%20Text-Prosody-Timbre%20Decomposition.pdf) for more details.

### Project Structure

```
stt-compress-tts/
├── src/
│   ├── stt/               # Speech-to-Text
│   ├── prosody/           # Prosody extraction
│   ├── speaker/           # Timbre/Speaker identification
│   ├── compression/       # Multi-stage compression
│   ├── tts/               # Text-to-Speech
│   ├── network/           # WebRTC & packet handling
│   ├── pipeline/          # Sender/receiver pipelines
│   ├── util/              # Util libs
│   ├── signaling/         # The signaling server
│   ├── audio/             # Audio transmission
│   └── app/               # CLI application
├── frontend/              # React web interface
├── tests/                 # Comprehensive test suite
├── benchmarks/            # Performance benchmarks
├── benchmarks_results/    # The benchmark results reported in the paper
├── configs/               # Quality mode configurations
├── plot_configs/          # Prosody Sampling Rate Analysis configurations
├── plots/                 # Prosody Sampling Rate Analysis result plots
├── paper/                 # Paper tex dev
├── weights/               # NISQA model weights
```

## Setup

Make sure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed first.

_Note that getting this project running is a bit complicated. You may need to manually edit the source code of some libraries to remove some false-positive exceptions raised by them. The dependency hell problem has already been resolved: some dependencies use my custom forks of them from my own forgejo instance, feel free to contact me if you cannot access them._ 

```bash
# Create venv
uv venv

# Run Unit Tests to auto-download the dependencies and models. Ideally, after manually resolving some libraries' problems, none should fail.
uv run pytest

# See the help screen of benchmarks module
uv run python -m benchmarks.run_all --help

# Run benchmark once: verify that everything is fine now 
uv run python -m benchmarks.run_all --audio <path/to/input/audio.wav> --reference <path/to/timbre/reference/audio.wav> --config balanced_mode
```

## Benchmarks

```bash
# Run the Prosody Sampling Rate Analysis
uv run python -m benchmarks.run_all --plot --librispeech 10000
uv run python -m benchmarks.run_all --plot-using-json benchmark_results.json

# Run the actual Benchmark
uv run python -m benchmarks.run_all --librispeech 10000 --noise high_quality --output test_librispeech.json
uv run python -m benchmarks.run_all --interpret test_librispeech.json
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** with additional restrictive terms.

For commercial licensing inquiries, please contact the project maintainers.

See the [LICENSE](LICENSE) file for complete terms.

## Acknowledgments

- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) - Fast STT
- [Coqui TTS](https://github.com/coqui-ai/TTS) - XTTS-v2 voice cloning
- [SpeechBrain](https://speechbrain.github.io/) - Timbre/Speaker embeddings
- [Parselmouth](https://github.com/YannickJadoul/Parselmouth) - Prosody extraction
- [aiortc](https://github.com/aiortc/aiortc) - WebRTC for Python
