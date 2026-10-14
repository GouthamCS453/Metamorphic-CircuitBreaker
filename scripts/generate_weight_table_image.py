import os
import textwrap
import matplotlib.pyplot as plt

def main():
    fig, ax = plt.subplots(figsize=(16, 7.8), dpi=300)
    ax.axis('off')

    plt.suptitle('Targeted Weight Ablation Configurations for (alpha, beta, gamma) in Research Papers', 
                 fontsize=16, fontweight='bold', y=0.96, color='#111827', fontfamily='sans-serif')
    plt.title('Evaluating Sensitivity and Robustness of the Composite Brittleness Index: CBI = alpha*max(IR_k) + beta*CFS + gamma*(1 - c0)', 
              fontsize=11.5, fontstyle='italic', y=1.01, color='#4B5563', fontfamily='sans-serif')

    columns = [
        'Configuration',
        'alpha\n(Peak Family)',
        'beta\n(Spread)',
        'gamma\n(Uncertainty)',
        'Scientific Hypothesis Tested',
        'Expected Outcome / Paper Insight'
    ]

    def wrap(text, width=38):
        return '\n'.join(textwrap.wrap(text, width=width))

    rows = [
        [
            '1. Proposed Balanced MCB\n(Standard Framework)',
            '0.50', '0.35', '0.15',
            wrap('Balances worst-family vulnerability, multi-family spread, and native model uncertainty.', 36),
            wrap('Optimal Youden J ~ 0.83: Best balance of high safety recall and low false alarms.', 36)
        ],
        [
            '2. Peak-Only Ablation\n(No Cross-Family Spread)',
            '0.85', '0.00', '0.15',
            wrap('"What if we ignore Cross-Family Spread (beta=0) and only care about the single worst family?"', 36),
            wrap('High False Alarms: A single noisy family (e.g. sensor blur) trips breaker even when rest is solid.', 36)
        ],
        [
            '3. Spread-Only Ablation\n(Spread-Dominant)',
            '0.15', '0.70', '0.15',
            wrap('"What if we only trip when failures spread across almost all families (beta=0.70)?"', 36),
            wrap('Lower Recall (Missed Errors): Fails to catch severe isolated weaknesses (e.g. blind to rotation).', 36)
        ],
        [
            '4. High-Confidence Ablation\n(High Softmax Trust)',
            '0.25', '0.25', '0.50',
            wrap('"What if we place heavy trust in the model\'s raw softmax confidence (gamma=0.50)?"', 36),
            wrap('Catastrophic Failure: Drops recall significantly because overconfident wrong guesses slip through.', 36)
        ],
        [
            '5. Pure Black-Box MT\n(Zero Softmax Trust)',
            '0.60', '0.40', '0.00',
            wrap('"What if we completely ignore model confidence and rely 100% on metamorphic testing?"', 36),
            wrap('Viable Alternative: Works well (J ~ 0.79), proving the metamorphic engine does the heavy lifting.', 36)
        ]
    ]

    col_widths = [0.22, 0.08, 0.08, 0.09, 0.27, 0.26]
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        colWidths=col_widths
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.3)

    header_color = '#1E3A8A'    # Deep Navy Blue
    alt_row_color_1 = '#F9FAFB' # Light Gray
    alt_row_color_2 = '#FFFFFF' # White
    proposed_color = '#EFF6FF'  # Soft Blue for row 1
    border_color = '#D1D5DB'

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(border_color)
        cell.set_linewidth(1.0)
        
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color='white', fontweight='bold', fontsize=10.5)
            cell.set_height(0.10)
        else:
            if c in [0, 4, 5]:
                cell.set_text_props(ha='left')
            else:
                cell.set_text_props(ha='center')
                
            if r == 1:
                cell.set_facecolor(proposed_color)
                if c == 0:
                    cell.set_text_props(fontweight='bold', color='#1E40AF')
            elif r % 2 == 1:
                cell.set_facecolor(alt_row_color_1)
            else:
                cell.set_facecolor(alt_row_color_2)
                
            cell.set_height(0.14)

    os.makedirs('Research Paper', exist_ok=True)
    os.makedirs('outputs/experiments/plots', exist_ok=True)

    out_path_1 = 'Research Paper/weight_ablation_table.jpg'
    out_path_2 = 'outputs/experiments/plots/weight_ablation_table.jpg'

    plt.savefig(out_path_1, format='jpg', dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.savefig(out_path_2, format='jpg', dpi=300, bbox_inches='tight', pad_inches=0.3)
    print(f'Saved: {out_path_1}')
    print(f'Saved: {out_path_2}')

if __name__ == '__main__':
    main()
