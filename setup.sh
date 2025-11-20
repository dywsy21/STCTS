#!/bin/bash
# Complete setup script for STT-Compression-TTS

set -e

echo "🎙️  STT-Compression-TTS Setup Script"
echo "===================================="
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo "❌ Python 3.10+ required. Found: $PYTHON_VERSION"
    exit 1
fi
echo "✅ Python $PYTHON_VERSION"

# Check if uv is installed
echo "Checking for uv..."
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo "✅ uv installed"
else
    echo "✅ uv found"
fi

# Check if Node.js is installed
echo "Checking for Node.js..."
if ! command -v node &> /dev/null; then
    echo "⚠️  Node.js not found. Frontend will not be available."
    echo "   Install Node.js 20+ from https://nodejs.org/"
    HAS_NODE=false
else
    NODE_VERSION=$(node --version | cut -d'v' -f2)
    echo "✅ Node.js $NODE_VERSION"
    HAS_NODE=true
fi

# Create virtual environment with uv
echo ""
echo "Creating virtual environment with uv..."
if [ ! -d .venv ]; then
    uv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment and install dependencies
echo ""
echo "Installing Python dependencies with uv..."
source .venv/bin/activate
uv pip install -r requirements.txt
echo "✅ Dependencies installed"

# Set up environment file
echo ""
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "✅ .env created. Please edit with your settings."
else
    echo "✅ .env already exists"
fi

# Download models
echo ""
echo "Downloading AI models (this may take 10-20 minutes)..."
echo "Models will be cached for future use."
echo ""
./scripts/setup_models.sh

# Install frontend dependencies
if [ "$HAS_NODE" = true ]; then
    echo ""
    echo "Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
    echo "✅ Frontend dependencies installed"
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p logs
mkdir -p data/audio
mkdir -p results
echo "✅ Directories created"

# Run a quick test
echo ""
echo "Running quick validation..."
python -c "from src.utils.config import load_config; print('✅ Config system works')"
python -c "import numpy as np; print('✅ NumPy works')"

# Summary
echo ""
echo "🎉 Setup Complete!"
echo ""
echo "Virtual environment created in .venv/"
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source .venv/bin/activate"
echo "2. Edit .env with your configuration"
echo "3. Start signaling server: python -m src.signaling.server"
echo "4. Start client: python -m src.app.cli client --peer-id your-id"
if [ "$HAS_NODE" = true ]; then
    echo "5. Start frontend: cd frontend && npm run dev"
fi
echo ""
echo "For detailed instructions, see docs/quickstart.md"
echo ""
echo "To test the pipeline without networking:"
echo "  python -m src.app.cli test-pipeline --input test.wav --output out.wav"
echo ""
echo "Happy calling! 🚀"
