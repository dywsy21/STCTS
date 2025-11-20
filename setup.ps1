# Complete setup script for STT-Compression-TTS (PowerShell)

Write-Host "🎙️  STT-Compression-TTS Setup Script" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "Checking Python version..."
try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python (\d+)\.(\d+)") {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
            Write-Host "❌ Python 3.10+ required. Found: $pythonVersion" -ForegroundColor Red
            exit 1
        }
        Write-Host "✅ $pythonVersion" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ Python not found. Please install Python 3.10+." -ForegroundColor Red
    exit 1
}

# Check if uv is installed
Write-Host "Checking for uv..."
try {
    $uvVersion = uv --version 2>&1
    Write-Host "✅ uv found" -ForegroundColor Green
} catch {
    Write-Host "❌ uv not found. Installing..." -ForegroundColor Yellow
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    # Refresh environment
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    Write-Host "✅ uv installed" -ForegroundColor Green
}

# Check if Node.js is installed
Write-Host "Checking for Node.js..."
try {
    $nodeVersion = node --version 2>&1
    Write-Host "✅ Node.js $nodeVersion" -ForegroundColor Green
    $hasNode = $true
} catch {
    Write-Host "⚠️  Node.js not found. Frontend will not be available." -ForegroundColor Yellow
    Write-Host "   Install Node.js 20+ from https://nodejs.org/"
    $hasNode = $false
}

# Create virtual environment with uv
Write-Host ""
Write-Host "Creating virtual environment with uv..."
if (-not (Test-Path .venv)) {
    uv venv
    Write-Host "✅ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "✅ Virtual environment already exists" -ForegroundColor Green
}

# Activate virtual environment and install dependencies
Write-Host ""
Write-Host "Installing Python dependencies with uv..."
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
Write-Host "✅ Dependencies installed" -ForegroundColor Green

# Set up environment file
Write-Host ""
if (-not (Test-Path .env)) {
    Write-Host "Creating .env file..."
    Copy-Item .env.example .env
    Write-Host "✅ .env created. Please edit with your settings." -ForegroundColor Green
} else {
    Write-Host "✅ .env already exists" -ForegroundColor Green
}

# Download models
Write-Host ""
Write-Host "Downloading AI models (this may take 10-20 minutes)..."
Write-Host "Models will be cached for future use."
Write-Host ""
.\scripts\setup_models.ps1

# Install frontend dependencies
if ($hasNode) {
    Write-Host ""
    Write-Host "Installing frontend dependencies..."
    Set-Location frontend
    npm install
    Set-Location ..
    Write-Host "✅ Frontend dependencies installed" -ForegroundColor Green
}

# Create necessary directories
Write-Host ""
Write-Host "Creating directories..."
New-Item -ItemType Directory -Force -Path logs | Out-Null
New-Item -ItemType Directory -Force -Path data\audio | Out-Null
New-Item -ItemType Directory -Force -Path results | Out-Null
Write-Host "✅ Directories created" -ForegroundColor Green

# Run a quick test
Write-Host ""
Write-Host "Running quick validation..."
python -c "from src.utils.config import load_config; print('✅ Config system works')"
python -c "import numpy as np; print('✅ NumPy works')"

# Summary
Write-Host ""
Write-Host "🎉 Setup Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Virtual environment created in .venv\"
Write-Host ""
Write-Host "To activate the environment:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Activate virtual environment: .\.venv\Scripts\Activate.ps1"
Write-Host "2. Edit .env with your configuration"
Write-Host "3. Start signaling server: python -m src.signaling.server"
Write-Host "4. Start client: python -m src.app.cli client --peer-id your-id"
if ($hasNode) {
    Write-Host "5. Start frontend: cd frontend; npm run dev"
}
Write-Host ""
Write-Host "For detailed instructions, see docs\quickstart.md"
Write-Host ""
Write-Host "To test the pipeline without networking:"
Write-Host "  python -m src.app.cli test-pipeline --input test.wav --output out.wav"
Write-Host ""
Write-Host "Happy calling! 🚀" -ForegroundColor Cyan
