#!/usr/bin/env python3
"""
Compare original and reconstructed prosody to evaluate compression quality.
Usage: python scripts/compare_prosody.py <original_audio.wav> <reconstructed_prosody.npz>
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def extract_original_prosody(audio_path: str):
    """Extract prosody from original audio."""
    import soundfile as sf
    import librosa
    from src.prosody import ProsodyExtractor
    
    # Load audio
    audio, sr = sf.read(audio_path)
    
    # Resample to 16kHz if needed
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    # Extract prosody
    extractor = ProsodyExtractor(sample_rate=sr)
    prosody = extractor.extract_all(audio)
    
    return prosody.pitch, prosody.energy, len(audio) / sr


def load_reconstructed_prosody(npz_path: str):
    """Load reconstructed prosody from npz file."""
    data = np.load(npz_path)
    return data['pitch'], data['energy']


def compare_prosody(original_pitch, original_energy, recon_pitch, recon_energy, duration, npz_path=""):
    """Compare and visualize original vs reconstructed prosody."""
    
    print("=" * 80)
    print("PROSODY COMPARISON")
    print("=" * 80)
    
    # Pitch comparison
    if original_pitch is not None and len(original_pitch) > 0 and len(recon_pitch) > 0:
        # Filter out zeros and NaNs for meaningful comparison
        orig_valid = original_pitch[(original_pitch > 0) & ~np.isnan(original_pitch)]
        recon_valid = recon_pitch[(recon_pitch > 0) & ~np.isnan(recon_pitch)]
        
        if len(orig_valid) > 0 and len(recon_valid) > 0:
            mae_pitch = np.mean(np.abs(orig_valid[:len(recon_valid)] - recon_valid[:len(orig_valid)]))
            rmse_pitch = np.sqrt(np.mean((orig_valid[:len(recon_valid)] - recon_valid[:len(orig_valid)])**2))
            corr_pitch = np.corrcoef(orig_valid[:len(recon_valid)], recon_valid[:len(orig_valid)])[0, 1]
            
            print(f"Pitch Analysis:")
            print(f"  Original: {len(original_pitch)} frames, mean={np.mean(orig_valid):.1f} Hz")
            print(f"  Reconstructed: {len(recon_pitch)} frames, mean={np.mean(recon_valid):.1f} Hz")
            print(f"  MAE: {mae_pitch:.2f} Hz")
            print(f"  RMSE: {rmse_pitch:.2f} Hz")
            print(f"  Correlation: {corr_pitch:.4f}")
    
    # Energy comparison
    if original_energy is not None and len(original_energy) > 0 and len(recon_energy) > 0:
        orig_valid_energy = original_energy[~np.isnan(original_energy)]
        recon_valid_energy = recon_energy[~np.isnan(recon_energy)]
        
        if len(orig_valid_energy) > 0 and len(recon_valid_energy) > 0:
            mae_energy = np.mean(np.abs(orig_valid_energy[:len(recon_valid_energy)] - recon_valid_energy[:len(orig_valid_energy)]))
            rmse_energy = np.sqrt(np.mean((orig_valid_energy[:len(recon_valid_energy)] - recon_valid_energy[:len(orig_valid_energy)])**2))
            corr_energy = np.corrcoef(orig_valid_energy[:len(recon_valid_energy)], recon_valid_energy[:len(orig_valid_energy)])[0, 1]
            
            print(f"\nEnergy Analysis:")
            print(f"  Original: {len(original_energy)} frames, mean={np.mean(orig_valid_energy):.4f}")
            print(f"  Reconstructed: {len(recon_energy)} frames, mean={np.mean(recon_valid_energy):.4f}")
            print(f"  MAE: {mae_energy:.6f}")
            print(f"  RMSE: {rmse_energy:.6f}")
            print(f"  Correlation: {corr_energy:.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot pitch
    if original_pitch is not None and len(original_pitch) > 0:
        time_orig = np.linspace(0, duration, len(original_pitch))
        time_recon = np.linspace(0, duration, len(recon_pitch))
        
        axes[0].plot(time_orig, original_pitch, 'b-', alpha=0.7, label='Original', linewidth=1)
        axes[0].plot(time_recon, recon_pitch, 'r--', alpha=0.7, label='Reconstructed', linewidth=1.5)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Pitch (Hz)')
        axes[0].set_title('Pitch Contour Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Plot energy
    if original_energy is not None and len(original_energy) > 0:
        time_orig = np.linspace(0, duration, len(original_energy))
        time_recon = np.linspace(0, duration, len(recon_energy))
        
        axes[1].plot(time_orig, original_energy, 'b-', alpha=0.7, label='Original', linewidth=1)
        axes[1].plot(time_recon, recon_energy, 'r--', alpha=0.7, label='Reconstructed', linewidth=1.5)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Energy (RMS)')
        axes[1].set_title('Energy Contour Comparison')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(npz_path).parent / f"{Path(npz_path).stem}_comparison.png"
    plt.savefig(output_path, dpi=150)
    print(f"\n📊 Comparison plot saved to: {output_path}")
    
    # Show plot
    plt.show()


def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/compare_prosody.py <original_audio.wav> <reconstructed_prosody.npz>")
        print("Example: python scripts/compare_prosody.py test_audios/tpo53-1.wav test_audios/tpo53-1_prosody_reconstructed.npz")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    npz_path = sys.argv[2]
    
    if not Path(audio_path).exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    if not Path(npz_path).exists():
        print(f"Error: Prosody file not found: {npz_path}")
        sys.exit(1)
    
    print("Extracting original prosody...")
    orig_pitch, orig_energy, duration = extract_original_prosody(audio_path)
    
    print("Loading reconstructed prosody...")
    recon_pitch, recon_energy = load_reconstructed_prosody(npz_path)
    
    compare_prosody(orig_pitch, orig_energy, recon_pitch, recon_energy, duration, npz_path)


if __name__ == "__main__":
    main()
