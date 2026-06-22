import pandas as pd, matplotlib.pyplot as plt, numpy as np

df = pd.read_csv('results/comparison.csv', index_col=0)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Ablation Study: Real vs Synthetic vs Mixed', fontsize=16, fontweight='bold')

colors = ['#3B82F6', '#F59E0B', '#10B981']
x = np.arange(len(df))

# mAP@0.5 bar chart
axes[0].bar(x, df['mAP50'], color=colors, width=0.5, edgecolor='white', linewidth=1.5)
axes[0].set_xticks(x); axes[0].set_xticklabels(df.index, fontsize=12)
axes[0].set_ylabel('mAP@0.5'); axes[0].set_title('mAP@0.5 Comparison')
axes[0].set_ylim(0, 1)
for i, v in enumerate(df['mAP50']):
    axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

# Precision / Recall grouped bar
w = 0.3
axes[1].bar(x - w/2, df['Precision'], width=w, label='Precision', color='#3B82F6')
axes[1].bar(x + w/2, df['Recall'], width=w, label='Recall', color='#10B981')
axes[1].set_xticks(x); axes[1].set_xticklabels(df.index, fontsize=12)
axes[1].set_ylabel('Score'); axes[1].set_title('Precision & Recall')
axes[1].legend(); axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('results/ablation_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print('Plot saved to results/ablation_comparison.png')
