import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_transmission_diagram():
    # Use a clean style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('ggplot') # Fallback

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    # Setup axes limits
    ax.set_xlim(-2, 10)
    ax.set_ylim(0, 4.5)
    
    # Hide standard axes
    ax.axis('off')
    
    # Colors
    c_text = '#2ca02c'    # Green
    c_prosody = '#1f77b4' # Blue
    c_timbre = '#ff7f0e'  # Orange
    c_loss = '#d62728'    # Red
    c_gray = '#808080'
    
    box_style = "round,pad=0.05,rounding_size=0.1"
    
    # --- 1. Timbre Stream (Top) ---
    y_timbre = 3.5
    # Dashed container
    rect_timbre = patches.FancyBboxPatch((0.2, y_timbre - 0.4), 9.6, 0.9, 
                                       boxstyle="round,pad=0.1", 
                                       linewidth=1, edgecolor=c_timbre, facecolor='none', linestyle='--')
    ax.add_patch(rect_timbre)
    ax.text(-1.0, y_timbre, 'Timbre Stream\n(Amortized)', ha='center', va='center', fontsize=11, fontweight='bold', color='#333333')

    # Initial Packet
    p_timbre = patches.FancyBboxPatch((0.5, y_timbre - 0.2), 1.2, 0.4, boxstyle=box_style, linewidth=1, edgecolor='black', facecolor=c_timbre, alpha=0.9)
    ax.add_patch(p_timbre)
    ax.text(1.1, y_timbre, 'Speaker\nEmbedding', ha='center', va='center', color='white', fontsize=8, fontweight='bold')
    
    # Long grey dashed arrow
    ax.annotate('', xy=(9.5, y_timbre), xytext=(2.0, y_timbre),
                arrowprops=dict(arrowstyle='->', color=c_gray, linestyle='dashed', linewidth=1.5))
    ax.text(5.75, y_timbre + 0.1, 'Sent Once (Amortized)', ha='center', va='bottom', color=c_gray, fontsize=10, style='italic', fontweight='bold')


    # --- 2. Text Stream (Middle) ---
    y_text = 2.2
    # Dashed container
    rect_text = patches.FancyBboxPatch((0.2, y_text - 0.4), 9.6, 0.9, 
                                     boxstyle="round,pad=0.1", 
                                     linewidth=1, edgecolor=c_text, facecolor='none', linestyle='--')
    ax.add_patch(rect_text)
    ax.text(-1.0, y_text, 'Text Stream\n(High Priority)', ha='center', va='center', fontsize=11, fontweight='bold', color='#333333')

    # Packet 1 (Success)
    p_text1 = patches.FancyBboxPatch((0.8, y_text - 0.2), 0.8, 0.4, boxstyle=box_style, linewidth=1, edgecolor='black', facecolor=c_text, alpha=0.9)
    ax.add_patch(p_text1)
    ax.text(1.2, y_text, 'Text 1', ha='center', va='center', color='white', fontsize=9, fontweight='bold')

    # Packet 2 (Loss)
    p_text2_loss = patches.FancyBboxPatch((3.5, y_text - 0.2), 0.8, 0.4, boxstyle=box_style, linewidth=2, edgecolor=c_loss, facecolor='white', linestyle='--')
    ax.add_patch(p_text2_loss)
    ax.text(3.9, y_text, 'Text 2', ha='center', va='center', color=c_loss, fontsize=9, fontweight='bold')
    # Red X
    ax.plot([3.5, 4.3], [y_text - 0.2, y_text + 0.2], color=c_loss, linewidth=2)
    ax.plot([3.5, 4.3], [y_text + 0.2, y_text - 0.2], color=c_loss, linewidth=2)
    
    # Retransmission Arrow (Curved)
    ax.annotate('High Priority: must be retransmitted', xy=(5.4, y_text + 0.25), xytext=(3.9, y_text + 0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=-0.4", color=c_text, lw=1.5),
                color=c_text, fontsize=9, ha='center', fontweight='bold')

    # Packet 2 (Retry Success)
    p_text2_retry = patches.FancyBboxPatch((5.0, y_text - 0.2), 0.8, 0.4, boxstyle=box_style, linewidth=1, edgecolor='black', facecolor=c_text, alpha=0.9)
    ax.add_patch(p_text2_retry)
    ax.text(5.4, y_text, 'Text 2\n(Retry)', ha='center', va='center', color='white', fontsize=9, fontweight='bold')

    # Packet 3 (Success)
    p_text3 = patches.FancyBboxPatch((8.0, y_text - 0.2), 0.8, 0.4, boxstyle=box_style, linewidth=1, edgecolor='black', facecolor=c_text, alpha=0.9)
    ax.add_patch(p_text3)
    ax.text(8.4, y_text, 'Text 3', ha='center', va='center', color='white', fontsize=9, fontweight='bold')


    # --- 3. Prosody Stream (Bottom) ---
    y_pros = 0.9
    # Dashed container
    rect_pros = patches.FancyBboxPatch((0.2, y_pros - 0.4), 9.6, 0.9, 
                                     boxstyle="round,pad=0.1", 
                                     linewidth=1, edgecolor=c_prosody, facecolor='none', linestyle='--')
    ax.add_patch(rect_pros)
    ax.text(-1.0, y_pros, 'Prosody Stream\n(Med/Low Priority)', ha='center', va='center', fontsize=11, fontweight='bold', color='#333333')

    # Prosody Packets (More frequent than text)
    # P1 (Keyframe - Medium Priority)
    p_pros1 = patches.FancyBboxPatch((0.8, y_pros - 0.2), 0.5, 0.4, boxstyle=box_style, linewidth=1, edgecolor='black', facecolor=c_prosody, alpha=0.9)
    ax.add_patch(p_pros1)
    ax.text(1.05, y_pros, 'Key1', ha='center', va='center', color='white', fontsize=9, fontweight='bold')

    # P2 (Delta - Low Priority)
    p_pros2 = patches.FancyBboxPatch((2.4, y_pros - 0.2), 0.5, 0.4, boxstyle=box_style, linewidth=1, edgecolor='black', facecolor=c_prosody, alpha=0.5)
    ax.add_patch(p_pros2)
    ax.text(2.65, y_pros, 'Delta1', ha='center', va='center', color='white', fontsize=9, fontweight='bold')

    # P3 (Delta Loss - Low Priority)
    p_pros3_loss = patches.FancyBboxPatch((4.0, y_pros - 0.2), 0.5, 0.4, boxstyle=box_style, linewidth=2, edgecolor=c_loss, facecolor='white', linestyle='--')
    ax.add_patch(p_pros3_loss)
    ax.text(4.25, y_pros, 'Delta2', ha='center', va='center', color=c_loss, fontsize=9, fontweight='bold')
    # Red X
    ax.plot([4.0, 4.5], [y_pros - 0.2, y_pros + 0.2], color=c_loss, linewidth=2)
    ax.plot([4.0, 4.5], [y_pros + 0.2, y_pros - 0.2], color=c_loss, linewidth=2)

    # Graceful Degradation Annotation
    ax.annotate('Delta Dropped\n(Graceful Degradation)', xy=(4.25, y_pros - 0.24), xytext=(4.25, y_pros - 0.6),
                arrowprops=dict(arrowstyle='->', color=c_prosody),
                color=c_prosody, fontsize=9, ha='center', fontweight='bold')
    
    # P4 (Delta - Low Priority)
    p_pros4 = patches.FancyBboxPatch((5.6, y_pros - 0.2), 0.5, 0.4, boxstyle=box_style, linewidth=1, edgecolor='black', facecolor=c_prosody, alpha=0.5)
    ax.add_patch(p_pros4)
    ax.text(5.85, y_pros, 'Delta3', ha='center', va='center', color='white', fontsize=9, fontweight='bold')

    # P5 (Keyframe Loss - Medium Priority)
    p_pros5_loss = patches.FancyBboxPatch((7.2, y_pros - 0.2), 0.5, 0.4, boxstyle=box_style, linewidth=2, edgecolor=c_loss, facecolor='white', linestyle='--')
    ax.add_patch(p_pros5_loss)
    ax.text(7.45, y_pros, 'Key2', ha='center', va='center', color=c_loss, fontsize=9, fontweight='bold')
    # Red X
    ax.plot([7.2, 7.7], [y_pros - 0.2, y_pros + 0.2], color=c_loss, linewidth=2)
    ax.plot([7.2, 7.7], [y_pros + 0.2, y_pros - 0.2], color=c_loss, linewidth=2)

    # Retransmission Arrow (Curved) for Keyframe
    ax.annotate('Keyframe: retransmit', xy=(8.8, y_pros + 0.25), xytext=(7.45, y_pros + 0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=-0.4", color=c_prosody, lw=1.5),
                color=c_prosody, fontsize=9, ha='center', fontweight='bold')

    # P6 (Keyframe Retry)
    p_pros6 = patches.FancyBboxPatch((8.8, y_pros - 0.2), 0.5, 0.4, boxstyle=box_style, linewidth=1, edgecolor='black', facecolor=c_prosody, alpha=0.9)
    ax.add_patch(p_pros6)
    ax.text(9.05, y_pros, 'Key2\n(Retry)', ha='center', va='center', color='white', fontsize=8, fontweight='bold')

    # --- 4. Time Axis ---
    ax.annotate('', xy=(9.8, 0.1), xytext=(0.2, 0.1),
                arrowprops=dict(arrowstyle='->', color='black', linewidth=1.5))
    ax.text(9.8, 0.0, 'Time', ha='right', va='top', fontsize=12)


    # Legend
    legend_elements = [
        patches.Patch(facecolor=c_text, edgecolor='black', label='Text (High Priority)'),
        patches.Patch(facecolor=c_prosody, alpha=0.9, edgecolor='black', label='Prosody Key (Medium Priority)'),
        patches.Patch(facecolor=c_prosody, alpha=0.5, edgecolor='black', label='Prosody Delta (Low Priority)'),
        patches.Patch(facecolor=c_timbre, edgecolor='black', label='Timbre (Amortized)'),
        patches.Patch(facecolor='white', edgecolor=c_loss, linestyle='--', linewidth=2, label='Packet Loss'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5, frameon=False, fontsize=10)

    plt.tight_layout()
    plt.savefig('transmission_mechanism.png')
    print("Plot generated: transmission_mechanism.png")

if __name__ == "__main__":
    draw_transmission_diagram()
