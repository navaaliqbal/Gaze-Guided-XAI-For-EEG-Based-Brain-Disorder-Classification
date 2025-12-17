import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

# Extract metrics from the training log
epochs = list(range(1, 36))
train_loss = [0.4544, 0.4239, 0.4082, 0.3643, 0.3828, 0.3214, 0.2960, 0.2283, 0.2097, 0.1965,
              0.2022, 0.1392, 0.1313, 0.1382, 0.0991, 0.0937, 0.1266, 0.0837, 0.1169, 0.0659,
              0.0731, 0.0832, 0.0545, 0.0639, 0.0621, 0.0718, 0.0645, 0.0400, 0.0362, 0.0516,
              0.0413, 0.0297, 0.0418, 0.0362, 0.0644]

train_acc = [68.80, 68.80, 70.24, 73.40, 69.70, 73.76, 76.01, 84.31, 84.76, 87.83,
             86.56, 91.79, 91.79, 91.16, 93.69, 94.14, 93.06, 94.50, 95.22, 95.85,
             96.48, 95.31, 96.84, 96.93, 96.48, 95.85, 97.29, 98.20, 98.29, 97.02,
             98.20, 98.29, 98.47, 98.20, 97.75]

eval_loss = [0.5542, 0.5237, 0.5024, 0.4769, 0.4047, 0.3197, 0.2492, 0.1855, 0.2968, 0.1787,
             0.3103, 0.1771, 0.1773, 0.2159, 0.3839, 0.1934, 0.1409, 0.1887, 0.2031, 0.1929,
             0.1918, 0.2751, 0.3076, 0.1598, 0.1675, 0.3211, 0.1408, 0.2398, 0.2692, 0.1765,
             0.1721, 0.1861, 0.1911, 0.2155, 0.2157]

eval_acc = [64.51, 64.51, 64.51, 64.51, 71.91, 82.10, 87.35, 89.20, 87.35, 87.96,
            88.58, 92.90, 92.59, 92.28, 89.81, 94.44, 95.06, 95.06, 93.52, 94.44,
            93.83, 92.90, 93.21, 95.06, 96.30, 93.21, 95.37, 95.37, 95.06, 96.60,
            96.30, 95.37, 95.68, 96.60, 95.99]

# Best model is at epoch 30
best_epoch = 30
best_eval_acc = eval_acc[best_epoch-1]
best_eval_loss = eval_loss[best_epoch-1]

# Create figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Training Progress Analysis', fontsize=16, fontweight='bold')

# Colors
train_color = '#1f77b4'  # Blue
eval_color = '#ff7f0e'   # Orange
best_color = '#2ca02c'   # Green
highlight_color = '#ffd700'  # Gold for highlighting

# 1. Loss curves
ax1.plot(epochs, train_loss, 'o-', color=train_color, linewidth=2, markersize=6, label='Train Loss')
ax1.plot(epochs, eval_loss, 's-', color=eval_color, linewidth=2, markersize=6, label='Eval Loss')
ax1.axvline(x=best_epoch, color=highlight_color, linestyle='--', linewidth=2, alpha=0.7, label=f'Best Model (Epoch {best_epoch})')
ax1.scatter(best_epoch, best_eval_loss, color=best_color, s=200, zorder=5, edgecolors='black', linewidth=2, marker='*', label=f'Best Eval Loss: {best_eval_loss:.4f}')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training and Evaluation Loss', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')
ax1.set_xticks(np.arange(1, 36, 2))

# 2. Accuracy curves
ax2.plot(epochs, train_acc, 'o-', color=train_color, linewidth=2, markersize=6, label='Train Acc')
ax2.plot(epochs, eval_acc, 's-', color=eval_color, linewidth=2, markersize=6, label='Eval Acc')
ax2.axvline(x=best_epoch, color=highlight_color, linestyle='--', linewidth=2, alpha=0.7, label=f'Best Model (Epoch {best_epoch})')
ax2.scatter(best_epoch, best_eval_acc, color=best_color, s=200, zorder=5, edgecolors='black', linewidth=2, marker='*', label=f'Best Eval Acc: {best_eval_acc:.2f}%')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Training and Evaluation Accuracy', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='lower right')
ax2.set_xticks(np.arange(1, 36, 2))

# 3. Zoomed accuracy (last 15 epochs)
zoom_start = 20
ax3.plot(epochs[zoom_start-1:], train_acc[zoom_start-1:], 'o-', color=train_color, linewidth=2, markersize=6, label='Train Acc')
ax3.plot(epochs[zoom_start-1:], eval_acc[zoom_start-1:], 's-', color=eval_color, linewidth=2, markersize=6, label='Eval Acc')
ax3.axvline(x=best_epoch, color=highlight_color, linestyle='--', linewidth=2, alpha=0.7, label=f'Best Model (Epoch {best_epoch})')
ax3.scatter(best_epoch, best_eval_acc, color=best_color, s=200, zorder=5, edgecolors='black', linewidth=2, marker='*')
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Accuracy (%)', fontsize=12)
ax3.set_title('Zoom: Accuracy (Epochs 21-35)', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(loc='lower right')
ax3.set_ylim([90, 100])  # Focus on high accuracy range
ax3.set_xticks(np.arange(zoom_start, 36, 2))

# 4. Summary table
summary_data = {
    'Metric': ['Best Epoch', 'Best Evaluation Accuracy', 'Best Evaluation Loss', 
               'Final Training Accuracy', 'Final Evaluation Accuracy',
               'Final Training Loss', 'Final Evaluation Loss',
               'Accuracy Improvement (Epoch 1 to Best)'],
    'Value': [f'{best_epoch}', f'{best_eval_acc:.2f}%', f'{best_eval_loss:.4f}',
              f'{train_acc[-1]:.2f}%', f'{eval_acc[-1]:.2f}%',
              f'{train_loss[-1]:.4f}', f'{eval_loss[-1]:.4f}',
              f'{(best_eval_acc - eval_acc[0]):.2f}%']
}

# Create a table in the last subplot
ax4.axis('tight')
ax4.axis('off')
table = ax4.table(cellText=list(zip(summary_data['Metric'], summary_data['Value'])),
                  colLabels=['Metric', 'Value'],
                  cellLoc='left',
                  loc='center',
                  colWidths=[0.6, 0.4])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

# Highlight the best model row
for i, key in enumerate(table.get_celld().keys()):
    cell = table.get_celld()[key]
    if key[0] == 0:  # Header row
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#2c3e50')
    elif key[1] == 0 and key[0] in [1, 2, 3]:  # Best model metrics
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#d4edda')  # Light green for best model

ax4.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
plt.show()

# Create a separate figure for combined view
fig2, (ax5, ax6) = plt.subplots(2, 1, figsize=(14, 10))

# Combined loss plot
ax5.plot(epochs, train_loss, 'o-', color=train_color, linewidth=2, markersize=6, label='Train Loss', alpha=0.8)
ax5.plot(epochs, eval_loss, 's-', color=eval_color, linewidth=2, markersize=6, label='Eval Loss', alpha=0.8)
ax5.axvline(x=best_epoch, color=highlight_color, linestyle='--', linewidth=2.5, alpha=0.9, label=f'Best Model (Epoch {best_epoch})')
# Highlight the best point
ax5.scatter(best_epoch, best_eval_loss, color=best_color, s=300, zorder=10, 
           edgecolors='black', linewidth=3, marker='*', label=f'Best Eval Loss: {best_eval_loss:.4f}')
# Add shaded area around best epoch
ax5.axvspan(best_epoch-0.5, best_epoch+0.5, alpha=0.2, color=highlight_color)
ax5.set_xlabel('Epoch', fontsize=12)
ax5.set_ylabel('Loss', fontsize=12)
ax5.set_title('Loss Progression with Best Model Highlighted', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend(loc='upper right')
ax5.set_xticks(np.arange(1, 36, 2))

# Combined accuracy plot
ax6.plot(epochs, train_acc, 'o-', color=train_color, linewidth=2, markersize=6, label='Train Accuracy', alpha=0.8)
ax6.plot(epochs, eval_acc, 's-', color=eval_color, linewidth=2, markersize=6, label='Eval Accuracy', alpha=0.8)
ax6.axvline(x=best_epoch, color=highlight_color, linestyle='--', linewidth=2.5, alpha=0.9, label=f'Best Model (Epoch {best_epoch})')
# Highlight the best point
ax6.scatter(best_epoch, best_eval_acc, color=best_color, s=300, zorder=10, 
           edgecolors='black', linewidth=3, marker='*', label=f'Best Eval Acc: {best_eval_acc:.2f}%')
# Add shaded area around best epoch
ax6.axvspan(best_epoch-0.5, best_epoch+0.5, alpha=0.2, color=highlight_color)
ax6.set_xlabel('Epoch', fontsize=12)
ax6.set_ylabel('Accuracy (%)', fontsize=12)
ax6.set_title('Accuracy Progression with Best Model Highlighted', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.legend(loc='lower right')
ax6.set_xticks(np.arange(1, 36, 2))

plt.tight_layout()
plt.show()

# Print key observations
print("="*80)
print("KEY OBSERVATIONS:")
print("="*80)
print(f"1. Best model achieved at epoch {best_epoch} with:")
print(f"   - Evaluation Accuracy: {best_eval_acc:.2f}%")
print(f"   - Evaluation Loss: {best_eval_loss:.4f}")
print(f"   - Balanced Accuracy: 96.98% (from classification report)")
print(f"   - Macro F1-Score: 0.9634")
print()
print(f"2. Overall improvement: {best_eval_acc - eval_acc[0]:.2f}% increase in accuracy")
print()
print(f"3. Training completed with:")
print(f"   - Final Training Accuracy: {train_acc[-1]:.2f}%")
print(f"   - Final Evaluation Accuracy: {eval_acc[-1]:.2f}%")
print(f"   - Training converged well with low loss values")
print()
print(f"4. Learning rate schedule:")
print(f"   - Epochs 1-23: 1.00e-04")
print(f"   - Epochs 24-35: 5.00e-05 (reduced at epoch 24)")