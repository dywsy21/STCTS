#!/bin/bash
# Bash script to test all config files and compare results
# Usage: ./scripts/test_all_configs.sh <audio_file>

set -e

if [ $# -eq 0 ]; then
    echo "Error: No audio file provided"
    echo "Usage: $0 <audio_file>"
    exit 1
fi

AUDIO_FILE="$1"

if [ ! -f "$AUDIO_FILE" ]; then
    echo "Error: Audio file not found: $AUDIO_FILE"
    exit 1
fi

# Config files to test
CONFIGS=("minimal_mode" "balanced_mode" "high_quality_mode")

echo "================================================================================"
echo "Testing audio file with all configurations"
echo "Audio: $AUDIO_FILE"
echo "================================================================================"
echo ""

# Store results
declare -A RESULTS

for config in "${CONFIGS[@]}"; do
    echo ""
    echo "▶ Testing with config: $config"
    echo "--------------------------------------------------------------------------------"
    
    # Run the test
    output=$(uv run python -m src.utils.cli test "$AUDIO_FILE" --config "$config" 2>&1)
    
    # Display output
    echo "$output"
    
    # Extract bitrate
    bitrate=$(echo "$output" | grep -oP 'Total:.*-> \K[\d.]+(?= bps)' || echo "N/A")
    RESULTS["$config"]="$bitrate"
    
    echo ""
done

# Display summary
echo ""
echo "================================================================================"
echo "SUMMARY - Bitrate Comparison"
echo "================================================================================"
printf "%-20s %s\n" "Config" "Bitrate (bps)"
echo "--------------------------------------------------------------------------------"

for config in "${CONFIGS[@]}"; do
    printf "%-20s %s\n" "$config" "${RESULTS[$config]}"
done

echo ""
echo "💡 Lower bitrate = better compression"
echo "💡 Reconstructed audio files saved with '_reconstructed.wav' suffix"
