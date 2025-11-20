#!/usr/bin/env python3
"""
Quick comparison tool to show differences between config files.
Usage: python scripts/compare_configs.py
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(path: Path) -> Dict[str, Any]:
    """Load a config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def print_comparison():
    """Print a comparison table of all config files."""
    configs_dir = Path("configs")
    config_files = [
        "minimal_mode.yaml",
        "balanced_mode.yaml", 
        "high_quality_mode.yaml"
    ]
    
    configs = {}
    for filename in config_files:
        path = configs_dir / filename
        if path.exists():
            configs[filename.replace("_mode.yaml", "")] = load_config(path)
    
    print("=" * 100)
    print("CONFIGURATION COMPARISON")
    print("=" * 100)
    print()
    
    # Compression settings
    print("📦 COMPRESSION SETTINGS")
    print("-" * 100)
    print(f"{'Setting':<40} {'Minimal':<20} {'Balanced':<20} {'High Quality':<20}")
    print("-" * 100)
    
    settings_to_compare = [
        ("Text Algorithm", "compression.text_algorithm"),
        ("Text Level", "compression.text_level"),
        ("Pitch Quantization (bits)", "compression.prosody_quantization_pitch_bits"),
        ("Energy Quantization (bits)", "compression.prosody_quantization_energy_bits"),
        ("Rate Quantization (bits)", "compression.prosody_quantization_rate_bits"),
        ("Delta Encoding", "compression.prosody_delta_encoding"),
        ("Timbre Algorithm", "compression.timbre_algorithm"),
    ]
    
    for label, key_path in settings_to_compare:
        values = []
        for mode in ["minimal", "balanced", "high_quality"]:
            if mode in configs:
                parts = key_path.split(".")
                value = configs[mode]
                for part in parts:
                    value = value.get(part, "N/A")
                values.append(str(value))
            else:
                values.append("N/A")
        print(f"{label:<40} {values[0]:<20} {values[1]:<20} {values[2]:<20}")
    
    print()
    
    # Network settings
    print("🌐 NETWORK SETTINGS")
    print("-" * 100)
    print(f"{'Setting':<40} {'Minimal':<20} {'Balanced':<20} {'High Quality':<20}")
    print("-" * 100)
    
    network_settings = [
        ("Max Bandwidth (bps)", "network.max_bandwidth_bps"),
        ("FEC Enabled", "network.fec_enabled"),
        ("Priority Queue", "network.priority_queue"),
    ]
    
    for label, key_path in network_settings:
        values = []
        for mode in ["minimal", "balanced", "high_quality"]:
            if mode in configs:
                parts = key_path.split(".")
                value = configs[mode]
                for part in parts:
                    value = value.get(part, "N/A")
                values.append(str(value))
            else:
                values.append("N/A")
        print(f"{label:<40} {values[0]:<20} {values[1]:<20} {values[2]:<20}")
    
    print()
    
    # Prosody settings
    print("🎵 PROSODY SETTINGS")
    print("-" * 100)
    print(f"{'Setting':<40} {'Minimal':<20} {'Balanced':<20} {'High Quality':<20}")
    print("-" * 100)
    
    prosody_settings = [
        ("Update Rate (Hz)", "prosody.update_rate_hz"),
        ("Features", "prosody.features"),
        ("Emotion Rate (Hz)", "prosody.emotion_rate_hz"),
    ]
    
    for label, key_path in prosody_settings:
        values = []
        for mode in ["minimal", "balanced", "high_quality"]:
            if mode in configs:
                parts = key_path.split(".")
                value = configs[mode]
                for part in parts:
                    value = value.get(part, "N/A")
                if isinstance(value, list):
                    value = ", ".join(value)
                values.append(str(value))
            else:
                values.append("N/A")
        print(f"{label:<40} {values[0]:<20} {values[1]:<20} {values[2]:<20}")
    
    print()
    
    # Model settings
    print("🤖 MODEL SETTINGS")
    print("-" * 100)
    print(f"{'Setting':<40} {'Minimal':<20} {'Balanced':<20} {'High Quality':<20}")
    print("-" * 100)
    
    model_settings = [
        ("STT Model", "stt.model"),
        ("STT Model Size", "stt.model_size"),
        ("TTS Model", "tts.model"),
    ]
    
    for label, key_path in model_settings:
        values = []
        for mode in ["minimal", "balanced", "high_quality"]:
            if mode in configs:
                parts = key_path.split(".")
                value = configs[mode]
                for part in parts:
                    value = value.get(part, "N/A")
                values.append(str(value))
            else:
                values.append("N/A")
        print(f"{label:<40} {values[0]:<20} {values[1]:<20} {values[2]:<20}")
    
    print()
    print("=" * 100)
    print()
    print("💡 Use these configs with: uv run python -m src.utils.cli test <audio> --config <mode>")
    print("   Example: uv run python -m src.utils.cli test audio.wav --config minimal_mode")
    print()


if __name__ == "__main__":
    print_comparison()
