"""
Quick test for plot generation with mock data.
"""

import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.plot_results import generate_all_plots

# Generate mock results
mock_results = []
prosody_rates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 
                 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

for i, rate in enumerate(prosody_rates):
    # Simulate the "prosody paradox" - lower rates = higher NISQA
    nisqa_base = 3.5 - (rate * 0.08)  # Decreases with rate
    nisqa_noise = np.random.normal(0, 0.05)
    nisqa_mos = np.clip(nisqa_base + nisqa_noise, 2.0, 4.5)
    
    # WER relatively constant (same STT model)
    wer = 0.25 + np.random.normal(0, 0.02)
    
    # Bitrate increases linearly with prosody rate
    text_bytes = 450
    prosody_bytes = int(50 + rate * 80)  # Grows with rate
    timbre_bytes = 100
    total_bytes = text_bytes + prosody_bytes + timbre_bytes
    duration = 10.0
    bitrate = (total_bytes * 8) / duration
    
    result = {
        'config': f'plot_{i+1:02d}_mode',
        'config_data': {
            'prosody': {
                'update_rate_hz': rate
            }
        },
        'reconstruction_success': True,
        'nisqa_mos': nisqa_mos,
        'nisqa_noisiness': 3.2 + np.random.normal(0, 0.1),
        'nisqa_coloration': 3.3 + np.random.normal(0, 0.1),
        'nisqa_discontinuity': 3.4 + np.random.normal(0, 0.1),
        'nisqa_loudness': 3.1 + np.random.normal(0, 0.1),
        'wer': wer,
        'speaker_similarity': 0.85 + np.random.normal(0, 0.02),
        'pesq': 2.5 + np.random.normal(0, 0.1),
        'stoi': 0.75 + np.random.normal(0, 0.02),
        'bitrate_bps': bitrate,
        'text_bytes': text_bytes,
        'prosody_bytes': prosody_bytes,
        'timbre_bytes': timbre_bytes,
    }
    
    mock_results.append(result)

# Generate plots
output_dir = Path("test_plots")
print(f"Generating test plots with {len(mock_results)} mock results...")
generate_all_plots(mock_results, output_dir)
print(f"\n✅ Test plots generated in: {output_dir.absolute()}")
print("Check the plots to verify they look beautiful! 🎨")
