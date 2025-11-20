"""
Beautiful visualization for benchmark results.

Generates publication-quality plots showing metrics vs parameters.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Set beautiful style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['grid.color'] = 'white'
plt.rcParams['grid.linewidth'] = 1.5
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def extract_plot_data(results: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Extract data from results for plotting.
    
    Args:
        results: List of benchmark results
        
    Returns:
        Dictionary with arrays of prosody rates and metrics
    """
    def get_prosody_rate(r):
        """Extract prosody rate from result, handling both dict and Config objects."""
        try:
            config_data = r.get('config_data')
            if config_data is None:
                return 0
            # Handle Config object (has prosody attribute)
            if hasattr(config_data, 'prosody'):
                prosody_obj = config_data.prosody
                return getattr(prosody_obj, 'update_rate_hz', 0)
            # Handle dict
            elif isinstance(config_data, dict):
                return config_data.get('prosody', {}).get('update_rate_hz', 0)
            else:
                return 0
        except Exception as e:
            logger.warning(f"Error extracting prosody rate: {e}")
            return 0
    
    # Sort by prosody rate
    sorted_results = sorted(results, key=get_prosody_rate)
    
    data = {
        'prosody_rate': [],
        'nisqa_mos': [],
        'nisqa_noisiness': [],
        'nisqa_coloration': [],
        'nisqa_discontinuity': [],
        'nisqa_loudness': [],
        'wer': [],
        'speaker_similarity': [],
        'pesq': [],
        'stoi': [],
        'bitrate_bps': [],
        'text_bps': [],
        'prosody_bps': [],
        'timbre_bps': [],
        'config_name': []
    }
    
    for result in sorted_results:
        # Skip if reconstruction failed
        if not result.get('reconstruction_success'):
            continue
        
        try:
            config_data = result.get('config_data')
            # Handle Config object (has prosody attribute)
            if hasattr(config_data, 'prosody'):
                prosody_obj = config_data.prosody
                prosody_rate = getattr(prosody_obj, 'update_rate_hz', 0)
            # Handle dict
            elif isinstance(config_data, dict):
                prosody_rate = config_data.get('prosody', {}).get('update_rate_hz', 0)
            else:
                prosody_rate = 0
            
            data['prosody_rate'].append(prosody_rate)
            data['nisqa_mos'].append(result.get('nisqa_mos', np.nan))
            data['nisqa_noisiness'].append(result.get('nisqa_noisiness', np.nan))
            data['nisqa_coloration'].append(result.get('nisqa_coloration', np.nan))
            data['nisqa_discontinuity'].append(result.get('nisqa_discontinuity', np.nan))
            data['nisqa_loudness'].append(result.get('nisqa_loudness', np.nan))
            data['wer'].append(result.get('wer', np.nan))
            data['speaker_similarity'].append(result.get('speaker_similarity', np.nan))
            data['pesq'].append(result.get('pesq', np.nan))
            data['stoi'].append(result.get('stoi', np.nan))
            data['bitrate_bps'].append(result.get('bitrate_bps', np.nan))
            data['text_bps'].append(result.get('text_bps', np.nan))
            data['prosody_bps'].append(result.get('prosody_bps', np.nan))
            data['timbre_bps'].append(result.get('timbre_bps', np.nan))
            data['config_name'].append(result.get('config', 'unknown'))
        except Exception as e:
            logger.warning(f"Error extracting data from result: {e}")
            continue
    
    # Convert to numpy arrays
    for key in data:
        if key != 'config_name':
            data[key] = np.array(data[key])
    
    return data


def plot_main_metrics(data: Dict[str, np.ndarray], output_path: Path):
    """Create main metrics plot (NISQA, WER, Bitrate).
    
    Args:
        data: Extracted plot data
        output_path: Output file path
    """
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    x = data['prosody_rate']
    
    # Color palette
    colors = {
        'nisqa': '#2ecc71',      # Green
        'wer': '#e74c3c',        # Red
        'similarity': '#3498db', # Blue
        'pesq': '#9b59b6',       # Purple
        'stoi': '#f39c12',       # Orange
        'bitrate': '#34495e'     # Dark gray
    }
    
    # 1. NISQA MOS (Main quality metric)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(x, data['nisqa_mos'], 'o-', color=colors['nisqa'], linewidth=2.5, 
             markersize=8, markeredgecolor='white', markeredgewidth=1.5, label='NISQA MOS')
    ax1.fill_between(x, data['nisqa_mos'], alpha=0.2, color=colors['nisqa'])
    
    # Add optimal region highlight
    optimal_idx = np.argmax(data['nisqa_mos'])
    optimal_rate = x[optimal_idx]
    ax1.axvline(optimal_rate, color=colors['nisqa'], linestyle='--', alpha=0.5, linewidth=2)
    ax1.annotate(f'Peak: {optimal_rate:.2f}Hz\nMOS: {data["nisqa_mos"][optimal_idx]:.3f}',
                xy=(optimal_rate, data['nisqa_mos'][optimal_idx]),
                xytext=(10, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=colors['nisqa'], linewidth=2),
                arrowprops=dict(arrowstyle='->', color=colors['nisqa'], linewidth=2),
                fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Prosody Update Rate (Hz)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('NISQA MOS Score', fontsize=12, fontweight='bold')
    ax1.set_title('🎯 Perceptual Quality vs Prosody Rate', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(x.min() - 0.5, x.max() + 0.5)
    ax1.set_ylim(data['nisqa_mos'].min() - 0.1, data['nisqa_mos'].max() + 0.2)
    
    # 2. WER (Intelligibility)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(x, data['wer'], 's-', color=colors['wer'], linewidth=2.5,
             markersize=7, markeredgecolor='white', markeredgewidth=1.5, label='WER')
    ax2.fill_between(x, data['wer'], alpha=0.2, color=colors['wer'])
    
    best_wer_idx = np.argmin(data['wer'])
    ax2.plot(x[best_wer_idx], data['wer'][best_wer_idx], 'o', 
             markersize=15, markerfacecolor='yellow', markeredgecolor=colors['wer'], 
             markeredgewidth=3, zorder=5)
    
    ax2.set_xlabel('Prosody Update Rate (Hz)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Word Error Rate', fontsize=11, fontweight='bold')
    ax2.set_title('📝 Intelligibility (Lower is Better)', fontsize=12, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(x.min() - 0.5, x.max() + 0.5)
    
    # 3. Speaker Similarity
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(x, data['speaker_similarity'], '^-', color=colors['similarity'], linewidth=2.5,
             markersize=7, markeredgecolor='white', markeredgewidth=1.5, label='Speaker Similarity')
    ax3.fill_between(x, data['speaker_similarity'], alpha=0.2, color=colors['similarity'])
    
    ax3.set_xlabel('Prosody Update Rate (Hz)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Speaker Similarity', fontsize=11, fontweight='bold')
    ax3.set_title('🎤 Voice Similarity', fontsize=12, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(x.min() - 0.5, x.max() + 0.5)
    ax3.set_ylim(0, 1.05)
    
    # 4. Bitrate
    ax4 = fig.add_subplot(gs[2, :])
    
    # Add component breakdown as stacked area
    ax4.stackplot(x, 
                   data['text_bps'],
                   data['prosody_bps'],
                   data['timbre_bps'],
                   labels=['Text', 'Prosody', 'Timbre'],
                   colors=['#95a5a6', '#e67e22', '#1abc9c'],
                   alpha=0.3)
                   
    # Plot total bitrate line on top
    ax4.plot(x, data['bitrate_bps'], 'D-', color=colors['bitrate'], linewidth=2.5,
             markersize=7, markeredgecolor='white', markeredgewidth=1.5, label='Total Bitrate')
    
    ax4.set_xlabel('Prosody Update Rate (Hz)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Bitrate (bps)', fontsize=12, fontweight='bold')
    ax4.set_title('📊 Bitrate Breakdown vs Prosody Rate', fontsize=14, fontweight='bold', pad=15)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(x.min() - 0.5, x.max() + 0.5)
    ax4.legend(loc='upper left', framealpha=0.9)
    
    # plt.suptitle('STT-Compress-TTS: Parameter Sweep Analysis', 
                #  fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"✅ Saved main metrics plot: {output_path}")
    plt.close()


def plot_nisqa_dimensions(data: Dict[str, np.ndarray], output_path: Path):
    """Create detailed NISQA dimensions plot.
    
    Args:
        data: Extracted plot data
        output_path: Output file path
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('NISQA Multi-Dimensional Quality Analysis', fontsize=16, fontweight='bold')
    
    x = data['prosody_rate']
    
    dimensions = [
        ('nisqa_mos', 'Overall MOS', '#2ecc71', 'o'),
        ('nisqa_noisiness', 'Noisiness', '#e74c3c', 's'),
        ('nisqa_coloration', 'Coloration', '#3498db', '^'),
        ('nisqa_discontinuity', 'Discontinuity', '#9b59b6', 'D'),
        ('nisqa_loudness', 'Loudness', '#f39c12', 'v'),
    ]
    
    for idx, (key, label, color, marker) in enumerate(dimensions):
        ax = axes.flat[idx]
        y = data[key]
        
        ax.plot(x, y, marker=marker, linestyle='-', color=color, linewidth=2.5,
                markersize=8, markeredgecolor='white', markeredgewidth=1.5)
        ax.fill_between(x, y, alpha=0.2, color=color)
        
        # Highlight extremes
        max_idx = np.argmax(y)
        min_idx = np.argmin(y)
        ax.plot(x[max_idx], y[max_idx], 'o', markersize=12, 
                markerfacecolor='lightgreen', markeredgecolor='green', markeredgewidth=2, zorder=5)
        ax.plot(x[min_idx], y[min_idx], 'o', markersize=12,
                markerfacecolor='lightcoral', markeredgecolor='red', markeredgewidth=2, zorder=5)
        
        ax.set_xlabel('Prosody Update Rate (Hz)', fontsize=10, fontweight='bold')
        ax.set_ylabel(f'{label} Score', fontsize=10, fontweight='bold')
        ax.set_title(f'{label}', fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
    
    # Sixth plot: Radar/spider chart summary
    ax = axes.flat[5]
    ax.axis('off')
    
    # # Create summary stats
    # summary_text = f"""
    # Summary Statistics
    
    # NISQA MOS Range: {data['nisqa_mos'].min():.3f} - {data['nisqa_mos'].max():.3f}
    # Optimal Rate: {x[np.argmax(data['nisqa_mos'])]:.2f} Hz
    
    # Quality Variance: {np.std(data['nisqa_mos']):.3f}
    # Bitrate Range: {data['bitrate_bps'].min():.0f} - {data['bitrate_bps'].max():.0f} bps
    # """
    
    # ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
    #         fontsize=11, verticalalignment='center', horizontalalignment='center',
    #         bbox=dict(boxstyle='round,pad=1', facecolor='#f0f0f0', edgecolor='#2ecc71', linewidth=3),
    #         family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"✅ Saved NISQA dimensions plot: {output_path}")
    plt.close()


def plot_quality_vs_bitrate(data: Dict[str, np.ndarray], output_path: Path):
    """Create quality-bitrate tradeoff scatter plot.
    
    Args:
        data: Extracted plot data
        output_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = data['bitrate_bps']
    y = data['nisqa_mos']
    colors_map = data['prosody_rate']
    
    # Create scatter plot with color gradient
    scatter = ax.scatter(x, y, c=colors_map, s=200, alpha=0.7, 
                        cmap='viridis', edgecolors='white', linewidth=2)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Prosody Update Rate (Hz)')
    cbar.ax.tick_params(labelsize=10)
    
    # Annotate some key points
    for i in [0, len(x)//4, len(x)//2, 3*len(x)//4, -1]:
        ax.annotate(f'{data["prosody_rate"][i]:.2f}Hz',
                   xy=(x[i], y[i]), xytext=(10, 10),
                   textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                   fontsize=9)
    
    # Add pareto frontier
    sorted_idx = np.argsort(x)
    pareto_y = np.maximum.accumulate(y[sorted_idx])
    ax.plot(x[sorted_idx], pareto_y, 'r--', linewidth=2, alpha=0.5, label='Pareto Frontier')
    
    # Highlight optimal point (best quality)
    best_idx = np.argmax(y)
    ax.plot(x[best_idx], y[best_idx], 'r*', markersize=25, 
            markeredgecolor='yellow', markeredgewidth=2, 
            label=f'Optimal: {data["prosody_rate"][best_idx]:.2f}Hz', zorder=10)
    
    ax.set_xlabel('Bitrate (bps)', fontsize=12, fontweight='bold')
    ax.set_ylabel('NISQA MOS Score', fontsize=12, fontweight='bold')
    ax.set_title('🎯 Quality-Bitrate Tradeoff Space', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    # Add efficiency annotation
    efficiency = y / (x / 1000)  # Quality per kbps
    best_efficiency_idx = np.argmax(efficiency)
    ax.plot(x[best_efficiency_idx], y[best_efficiency_idx], 'g*', markersize=20,
            markeredgecolor='white', markeredgewidth=2,
            label=f'Most Efficient: {data["prosody_rate"][best_efficiency_idx]:.2f}Hz', zorder=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"✅ Saved quality-bitrate plot: {output_path}")
    plt.close()


def plot_perceptual_metrics(data: Dict[str, np.ndarray], output_path: Path):
    """Create comprehensive perceptual metrics comparison.
    
    Args:
        data: Extracted plot data
        output_path: Output file path
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Perceptual Quality Metrics Comparison', fontsize=16, fontweight='bold')
    
    x = data['prosody_rate']
    
    metrics = [
        ('nisqa_mos', 'NISQA MOS (Naturalness)', '#2ecc71', 'o', (1, 5)),
        ('pesq', 'PESQ (Perceptual Quality)', '#9b59b6', 's', (-0.5, 4.5)),
        ('stoi', 'STOI (Intelligibility)', '#f39c12', '^', (0, 1)),
        ('speaker_similarity', 'Speaker Similarity', '#3498db', 'D', (0, 1)),
    ]
    
    for idx, (key, label, color, marker, ylim) in enumerate(metrics):
        ax = axes.flat[idx]
        y = data[key]
        
        ax.plot(x, y, marker=marker, linestyle='-', color=color, linewidth=2.5,
                markersize=8, markeredgecolor='white', markeredgewidth=1.5)
        ax.fill_between(x, y, alpha=0.2, color=color)
        
        # Add trend line
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_smooth, p(x_smooth), '--', color=color, alpha=0.5, linewidth=2)
        
        ax.set_xlabel('Prosody Update Rate (Hz)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Score', fontsize=10, fontweight='bold')
        ax.set_title(label, fontsize=11, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
        if ylim:
            ax.set_ylim(ylim)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"✅ Saved perceptual metrics plot: {output_path}")
    plt.close()


def generate_all_plots(results: List[Dict[str, Any]], output_dir: Path):
    """Generate all plot types.
    
    Args:
        results: List of benchmark results
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📊 Generating beautiful plots from {len(results)} results...")
    
    # Extract data
    data = extract_plot_data(results)
    
    if len(data['prosody_rate']) == 0:
        logger.error("❌ No valid data to plot!")
        return
    
    logger.info(f"📈 Plotting {len(data['prosody_rate'])} data points...")
    
    # Generate plots
    plot_main_metrics(data, output_dir / 'main_metrics.png')
    plot_nisqa_dimensions(data, output_dir / 'nisqa_dimensions.png')
    plot_quality_vs_bitrate(data, output_dir / 'quality_bitrate_tradeoff.png')
    plot_perceptual_metrics(data, output_dir / 'perceptual_metrics.png')
    
    logger.info(f"✅ All plots saved to: {output_dir}")
    logger.info("📊 Generated plots:")
    logger.info(f"  - main_metrics.png (Primary analysis)")
    logger.info(f"  - nisqa_dimensions.png (Detailed NISQA breakdown)")
    logger.info(f"  - quality_bitrate_tradeoff.png (Efficiency analysis)")
    logger.info(f"  - perceptual_metrics.png (All perceptual metrics)")


if __name__ == "__main__":
    # Standalone mode - load results from JSON and plot
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m benchmarks.plot_results <results.json> [output_dir]")
        sys.exit(1)
    
    results_file = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("plots")
    
    with open(results_file) as f:
        results = json.load(f)
    
    generate_all_plots(results, output_dir)
