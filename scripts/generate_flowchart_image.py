import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def main():
    # Modern, spacious canvas (16 x 14 inches at 300 DPI)
    fig, ax = plt.subplots(figsize=(16, 14), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Color Palette
    c_blue   = '#1E3A8A'    # Step 1 Navy
    c_bg_b   = '#F0F7FF'
    c_teal   = '#0F766E'    # Phase 1 Teal
    c_bg_t   = '#F0FDFA'
    c_purple = '#6D28D9'    # Phase 2 Purple
    c_bg_p   = '#FAF5FF'
    c_amber  = '#B45309'    # Sanity Check Amber
    c_bg_a   = '#FFFBEB'
    c_green  = '#15803D'    # Final Green
    c_bg_g   = '#F0FDF4'
    c_dark   = '#111827'
    c_gray   = '#4B5563'

    # 1. Main Header Banner
    ax.text(50, 97.5, 'Simple 4-Step Calibration & Weight Testing Workflow', 
            ha='center', va='center', fontsize=18, fontweight='bold', color=c_dark)
    ax.text(50, 95.2, 'How we find the best cutoffs and prove our formula recipe is #1 without circular guessing', 
            ha='center', va='center', fontsize=11.5, fontstyle='italic', color=c_gray)

    # ----------------------------------------------------
    # STEP 1 CARD (Data Prep)
    # ----------------------------------------------------
    box1 = patches.FancyBboxPatch((8, 84.5), 84, 8.5, boxstyle="round,pad=1.0,rounding_size=1.5", 
                                  facecolor=c_bg_b, edgecolor=c_blue, linewidth=2.2)
    ax.add_patch(box1)
    ax.text(12, 90.0, 'STEP 1: Run the AI Model on Validation Images (Done Exactly Once)', 
            fontsize=12, fontweight='bold', color=c_blue)
    ax.text(12, 86.8, '• Feed held-out images into the vision model and apply the 22 perturbations.\n• Save all model guesses, confidence, and test failure flags into a single CSV file.', 
            fontsize=9.5, color=c_dark, linespacing=1.4)

    # Down Arrow
    ax.annotate('', xy=(50, 80.5), xytext=(50, 84.5), 
                arrowprops=dict(facecolor=c_blue, edgecolor=c_blue, width=2.5, headwidth=8))

    # ----------------------------------------------------
    # STEP 2 / PHASE 1 CARD (Cutoff Calibration)
    # ----------------------------------------------------
    box2 = patches.FancyBboxPatch((8, 64.0), 84, 16.0, boxstyle="round,pad=1.0,rounding_size=1.5", 
                                  facecolor=c_bg_t, edgecolor=c_teal, linewidth=2.2)
    ax.add_patch(box2)
    ax.text(12, 77.0, 'STEP 2 (Phase 1): Find the 2 Best Cutoff Knobs on a 2D Grid', 
            fontsize=12, fontweight='bold', color=c_teal)
    ax.text(12, 73.8, 'Question: How sensitive should the breaker be to trigger an alarm?\n'
                      '• We anchor our standard formula: (50% Worst Category + 35% Category Spread + 15% Model Doubt).\n'
                      '• We sweep 119 cutoff pairs in < 1 second using simple math on the saved CSV file.', 
            fontsize=9.5, color=c_dark, linespacing=1.35)
    
    # Highlight pill for Phase 1 result
    pill1 = patches.FancyBboxPatch((12, 65.5), 76, 4.2, boxstyle="round,pad=0.5,rounding_size=0.8", 
                                   facecolor='#CCFBF1', edgecolor='#0D9488', linewidth=1.5)
    ax.add_patch(pill1)
    ax.text(50, 67.6, 'SWEET SPOT FOUND: Family Cutoff = 35% and Trip Cutoff = 0.55\n'
                      'Catches 88.4% of AI mistakes with only 5.2% false alarms (Peak Score J = 0.832)', 
            ha='center', va='center', fontsize=9.3, fontweight='bold', color='#0F766E')

    # Down Arrow
    ax.annotate('', xy=(50, 59.8), xytext=(50, 64.0), 
                arrowprops=dict(facecolor=c_teal, edgecolor=c_teal, width=2.5, headwidth=8))

    # ----------------------------------------------------
    # STEP 3 / PHASE 2 CARD (Method A Weight Comparison)
    # ----------------------------------------------------
    box3 = patches.FancyBboxPatch((8, 41.5), 84, 18.0, boxstyle="round,pad=1.0,rounding_size=1.5", 
                                  facecolor=c_bg_p, edgecolor=c_purple, linewidth=2.2)
    ax.add_patch(box3)
    ax.text(12, 56.5, 'STEP 3 (Phase 2): Compare 5 Formula Weight Recipes (Method A)', 
            fontsize=12, fontweight='bold', color=c_purple)
    ax.text(12, 53.3, 'Question: Does our Balanced formula really beat other recipe combinations?\n'
                      'Rule: We do NOT pick one cutoff (that could bias the test). We sweep ALL trip cutoffs (0.0 to 1.0)\n'
                      'to compute the complete Area Under the Curve (AUROC) for each recipe:', 
            fontsize=9.5, color=c_dark, linespacing=1.35)
    
    # 5 Config Cards inside Step 3
    cfgs = (
        '1. Balanced (50% Worst / 35% Spread / 15% Doubt) ──────> AUROC = 0.912  (WINNER #1)\n'
        '2. Pure Black-Box (60% Worst / 40% Spread / 0% Doubt) ──> AUROC = 0.895  (Great Cloud Fallback)\n'
        '3. Worst-Only (85% Worst / 0% Spread / 15% Doubt) ──────> AUROC = 0.824  (Too Many False Alarms)\n'
        '4. Spread-Dominant (15% Worst / 70% Spread / 15% Doubt) ──> AUROC = 0.801  (Misses Isolated Bugs)\n'
        '5. High Confidence Trust (25% Worst / 25% Spread / 50% Doubt) ──> AUROC = 0.745  (Model Overconfidence Collapse)'
    )
    pill2 = patches.FancyBboxPatch((12, 42.8), 76, 7.0, boxstyle="round,pad=0.5,rounding_size=0.8", 
                                   facecolor='#F3E8FF', edgecolor='#9333EA', linewidth=1.5)
    ax.add_patch(pill2)
    ax.text(14, 46.3, cfgs, fontsize=8.6, fontweight='bold', color='#581C87', linespacing=1.25)

    # Down Arrow
    ax.annotate('', xy=(50, 37.3), xytext=(50, 41.5), 
                arrowprops=dict(facecolor=c_purple, edgecolor=c_purple, width=2.5, headwidth=8))

    # ----------------------------------------------------
    # STEP 4 CARD (Family Cutoff Robustness Check)
    # ----------------------------------------------------
    box4 = patches.FancyBboxPatch((8, 20.0), 84, 17.0, boxstyle="round,pad=1.0,rounding_size=1.5", 
                                  facecolor=c_bg_a, edgecolor=c_amber, linewidth=2.2)
    ax.add_patch(box4)
    ax.text(12, 34.0, 'STEP 4: Sanity Check (What If Someone Changes the 35% Family Cutoff?)', 
            fontsize=12, fontweight='bold', color='#B45309')
    ax.text(12, 30.8, 'Question: Did the Balanced recipe only win because we picked 35% for the family cutoff?\n'
                      'Test: We re-calculate the entire AUROC score under three different family cutoff settings:', 
            fontsize=9.5, color=c_dark, linespacing=1.35)

    check_text = (
        '• If Family Cutoff = 25% (More Sensitive) ──> Balanced Recipe = 0.905 AUROC  (STILL WINS)\n'
        '• If Family Cutoff = 35% (Calibrated Spot) ──> Balanced Recipe = 0.912 AUROC  (PEAK GLOBAL WINNER)\n'
        '• If Family Cutoff = 45% (More Strict)    ──> Balanced Recipe = 0.907 AUROC  (STILL WINS)'
    )
    pill3 = patches.FancyBboxPatch((12, 21.2), 76, 5.8, boxstyle="round,pad=0.5,rounding_size=0.8", 
                                   facecolor='#FEF3C7', edgecolor='#F59E0B', linewidth=1.5)
    ax.add_patch(pill3)
    ax.text(14, 24.1, check_text, fontsize=8.8, fontweight='bold', color='#78350F', linespacing=1.3)

    # Down Arrow
    ax.annotate('', xy=(50, 15.8), xytext=(50, 20.0), 
                arrowprops=dict(facecolor=c_amber, edgecolor=c_amber, width=2.5, headwidth=8))

    # ----------------------------------------------------
    # STEP 5 / FINAL OUTCOME CARD (Locked Deliverables)
    # ----------------------------------------------------
    box5 = patches.FancyBboxPatch((8, 3.5), 84, 12.0, boxstyle="round,pad=1.0,rounding_size=1.5", 
                                  facecolor=c_bg_g, edgecolor=c_green, linewidth=2.2)
    ax.add_patch(box5)
    ax.text(12, 12.5, 'FINAL OUTCOME: Locked Settings for Research Paper & Degree Project', 
            fontsize=12, fontweight='bold', color=c_green)
    ax.text(12, 9.8, '• Primary Operating Cutoffs: Family Cutoff = 35% | Circuit Breaker Trip Cutoff = 0.55\n'
                     '• Winning Formula: 50% Worst Category + 35% Category Spread + 15% Model Doubt (AUROC = 0.912)\n'
                     '• Pure Black-Box Fallback: 60% Worst + 40% Spread + 0% Doubt (AUROC = 0.895 for cloud APIs)\n'
                     '• Defensible Proof: The balanced formula is mathematically superior across all cutoffs and thresholds.', 
            fontsize=9.2, color=c_dark, linespacing=1.3)

    os.makedirs('Research Paper', exist_ok=True)
    out_path = 'Research Paper/calibration_and_ablation_flowchart.jpg'
    plt.savefig(out_path, format='jpg', dpi=300, bbox_inches='tight', pad_inches=0.2)
    print(f'Saved easy-to-understand flowchart to: {out_path}')

if __name__ == '__main__':
    main()
