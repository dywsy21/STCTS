# Setup script for downloading and preparing models (PowerShell version)

Write-Host "=== STT-Compression-TTS Model Setup ===" -ForegroundColor Cyan
Write-Host ""

# Create models directory
$MODELS_DIR = ".\models"
New-Item -ItemType Directory -Force -Path $MODELS_DIR | Out-Null

Write-Host "Models will be downloaded to: $MODELS_DIR"
Write-Host ""

# Faster-Whisper models
Write-Host "[1/4] Downloading Faster-Whisper model..." -ForegroundColor Yellow
Write-Host "Note: Faster-Whisper models are downloaded automatically on first use"
Write-Host "Model: distil-large-v3 (~1.5 GB)"
python -c "from faster_whisper import WhisperModel; model = WhisperModel('distil-large-v3', device='cpu', compute_type='int8')"
Write-Host "✓ Faster-Whisper model ready" -ForegroundColor Green
Write-Host ""

# SpeechBrain models  
Write-Host "[2/4] Downloading SpeechBrain speaker embedding model..." -ForegroundColor Yellow
python -c @"
import torch
from speechbrain.pretrained import EncoderClassifier
classifier = EncoderClassifier.from_hparams(
    source='speechbrain/spkrec-ecapa-voxceleb',
    savedir='$MODELS_DIR/speechbrain/spkrec-ecapa-voxceleb'
)
print('✓ SpeechBrain speaker model ready')
"@
Write-Host ""

# Coqui TTS models
Write-Host "[3/4] Downloading Coqui TTS (XTTS-v2) model..." -ForegroundColor Yellow
Write-Host "Note: XTTS-v2 is large (~2 GB), this may take a while..."
python -c @"
from TTS.api import TTS
tts = TTS(model_name='tts_models/multilingual/multi-dataset/xtts_v2', progress_bar=True)
print('✓ XTTS-v2 model ready')
"@
Write-Host ""

# Emotion recognition model
Write-Host "[4/4] Downloading emotion recognition model..." -ForegroundColor Yellow
python -c @"
from speechbrain.pretrained import EncoderClassifier
classifier = EncoderClassifier.from_hparams(
    source='speechbrain/emotion-recognition-wav2vec2-IEMOCAP',
    savedir='$MODELS_DIR/speechbrain/emotion-recognition'
)
print('✓ Emotion recognition model ready')
"@
Write-Host ""

Write-Host "=== Setup Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "All models have been downloaded and are ready to use!" -ForegroundColor Green
Write-Host ""
Write-Host "You can now run the application:"
Write-Host "  poetry run stt-tts --help"
Write-Host ""
