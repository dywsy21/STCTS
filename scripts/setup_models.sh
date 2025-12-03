#!/bin/bash
# Setup script for downloading and preparing models

set -e

echo "=== STT-Compression-TTS Model Setup ==="
echo ""

# Create models directory
MODELS_DIR="./models"
mkdir -p "$MODELS_DIR"

echo "Models will be downloaded to: $MODELS_DIR"
echo ""

# Faster-Whisper models
echo "[1/4] Downloading Faster-Whisper model..."
echo "Note: Faster-Whisper models are downloaded automatically on first use"
echo "Model: distil-large-v3 (~1.5 GB)"
python -c "from faster_whisper import WhisperModel; model = WhisperModel('distil-large-v3', device='cpu', compute_type='int8')"
echo "✓ Faster-Whisper model ready"
echo ""

# SpeechBrain models  
echo "[2/4] Downloading SpeechBrain speaker embedding model..."
python -c "
import torch
from speechbrain.pretrained import EncoderClassifier
classifier = EncoderClassifier.from_hparams(
    source='speechbrain/spkrec-ecapa-voxceleb',
    savedir='$MODELS_DIR/speechbrain/spkrec-ecapa-voxceleb'
)
print('✓ SpeechBrain speaker model ready')
"
echo ""

# Coqui TTS models
echo "[3/4] Downloading Coqui TTS (XTTS-v2) model..."
echo "Note: XTTS-v2 is large (~2 GB), this may take a while..."
python -c "
from TTS.api import TTS
tts = TTS(model_name='tts_models/multilingual/multi-dataset/xtts_v2', progress_bar=True)
print('✓ XTTS-v2 model ready')
"
echo ""

# Emotion recognition model
echo "[4/4] Downloading emotion recognition model..."
python -c "
from speechbrain.pretrained import EncoderClassifier
classifier = EncoderClassifier.from_hparams(
    source='speechbrain/emotion-recognition-wav2vec2-IEMOCAP',
    savedir='$MODELS_DIR/speechbrain/emotion-recognition'
)
print('✓ Emotion recognition model ready')
"
echo ""

echo "=== Setup Complete ==="
echo ""
echo "All models have been downloaded and are ready to use!"
echo ""
echo "Directory structure:"
tree -L 2 "$MODELS_DIR" 2>/dev/null || ls -R "$MODELS_DIR"
echo ""
echo "You can now run the application:"
echo "  poetry run stt-tts --help"
echo ""
