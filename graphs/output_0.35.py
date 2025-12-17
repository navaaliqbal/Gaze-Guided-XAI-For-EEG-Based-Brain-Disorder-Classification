import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set style for clean plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create figure with 2x3 grid
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.25)

# ========== PLOT 1: ACCURACY PROGRESSION ==========
ax1 = fig.add_subplot(gs[0, 0])
epochs = list(range(1, 31))
train_acc = [55.37, 50.68, 54.28, 57.44, 61.77, 57.08, 58.70, 60.96, 63.03, 67.09,
             65.55, 63.30, 68.98, 71.15, 71.33, 74.93, 71.42, 74.57, 72.50, 74.93,
             74.66, 76.47, 75.65, 75.38, 75.83, 73.76, 74.84, 75.38, 75.83, 75.56]
eval_acc = [47.22, 55.86, 56.79, 59.57, 58.64, 63.27, 66.05, 63.58, 63.89, 66.67,
            67.59, 66.67, 66.36, 66.98, 66.36, 72.22, 69.44, 71.60, 72.53, 72.53,
            71.30, 71.30, 71.60, 71.30, 72.22, 71.91, 70.68, 71.60, 70.99, 71.60]

ax1.plot(epochs, train_acc, 'o-', linewidth=2.5, markersize=6, label='Train Accuracy', color='#2E86AB')
ax1.plot(epochs, eval_acc, 's-', linewidth=2.5, markersize=6, label='Eval Accuracy', color='#A23B72')

# Highlight best evaluation accuracy
best_epoch = epochs[eval_acc.index(max(eval_acc))]
best_acc = max(eval_acc)
ax1.plot(best_epoch, best_acc, 'r*', markersize=15, label=f'Best: {best_acc:.1f}%', zorder=5)
ax1.annotate(f'Epoch {best_epoch}: {best_acc:.1f}%', 
             xy=(best_epoch, best_acc), 
             xytext=(best_epoch+1, best_acc+2),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=11, fontweight='bold', color='red')

ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Training vs Evaluation Accuracy', fontsize=14, fontweight='bold', pad=15)
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 31)
ax1.set_ylim(40, 80)

# ========== PLOT 2: LOSS PROGRESSION ==========
ax2 = fig.add_subplot(gs[0, 1])
train_loss = [0.7053, 0.6973, 0.7029, 0.6906, 0.6880, 0.6878, 0.6884, 0.6751, 0.6718, 0.6587,
              0.6618, 0.6655, 0.6254, 0.6369, 0.6085, 0.5823, 0.5934, 0.5615, 0.5880, 0.5657,
              0.5695, 0.5540, 0.5529, 0.5593, 0.5465, 0.5754, 0.5579, 0.5594, 0.5589, 0.5585]
eval_loss = [0.7277, 0.7139, 0.7025, 0.6798, 0.6932, 0.6787, 0.6699, 0.6702, 0.6634, 0.6375,
             0.6339, 0.6399, 0.6298, 0.6527, 0.6338, 0.5722, 0.6077, 0.6004, 0.5962, 0.5924,
             0.5941, 0.5991, 0.5914, 0.5953, 0.5869, 0.5909, 0.5955, 0.5917, 0.5964, 0.5887]

ax2.plot(epochs, train_loss, 'o-', linewidth=2.5, markersize=6, label='Train Loss', color='#2E86AB')
ax2.plot(epochs, eval_loss, 's-', linewidth=2.5, markersize=6, label='Eval Loss', color='#A23B72')

# Find minimum loss point
min_eval_loss_idx = eval_loss.index(min(eval_loss))
ax2.plot(epochs[min_eval_loss_idx], min(eval_loss), 'g*', markersize=15, 
         label=f'Min Eval: {min(eval_loss):.4f}', zorder=5)

ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax2.set_title('Training vs Evaluation Loss', fontsize=14, fontweight='bold', pad=15)
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 31)

# ========== PLOT 3: LEARNING RATE SCHEDULE ==========
ax3 = fig.add_subplot(gs[0, 2])
learning_rates = [1e-4]*17 + [1e-5]*5 + [1e-6]*3 + [1e-7]*3 + [1e-8]*2

ax3.plot(epochs, learning_rates, 'D-', linewidth=2.5, markersize=8, color='#F18F01')
ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax3.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold', pad=15)
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3, which='both')
ax3.set_xlim(0, 31)

# Annotate schedule changes
schedule_points = [(17, '10x↓'), (22, '10x↓'), (25, '10x↓'), (28, '10x↓')]
for epoch, label in schedule_points:
    ax3.annotate(label, xy=(epoch, learning_rates[epoch-1]), 
                 xytext=(epoch, learning_rates[epoch-1]*3),
                 arrowprops=dict(arrowstyle='->', color='gray', lw=1),
                 fontsize=10, ha='center')

# ========== PLOT 4: CLASS IMBALANCE ANALYSIS ==========
ax4 = fig.add_subplot(gs[1, 0])
class_counts = {'Class 0': 209, 'Class 1': 115}
colors = ['#2E86AB', '#A23B72']

bars = ax4.bar(class_counts.keys(), class_counts.values(), color=colors, alpha=0.8)
ax4.set_xlabel('Class', fontsize=12, fontweight='bold')
ax4.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
ax4.set_title('Class Distribution (Evaluation Set)', fontsize=14, fontweight='bold', pad=15)
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, count in zip(bars, class_counts.values()):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{count}\n({count/324*100:.1f}%)',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

ax4.set_ylim(0, 250)

# ========== PLOT 5: PRECISION-RECALL BY CLASS ==========
ax5 = fig.add_subplot(gs[1, 1])
final_precision = [0.7773, 0.6018]
final_recall = [0.7847, 0.5913]
x = np.arange(2)
width = 0.35

bars1 = ax5.bar(x - width/2, final_precision, width, label='Precision', color='#2E86AB', alpha=0.8)
bars2 = ax5.bar(x + width/2, final_recall, width, label='Recall', color='#A23B72', alpha=0.8)

ax5.set_xlabel('Class', fontsize=12, fontweight='bold')
ax5.set_ylabel('Score', fontsize=12, fontweight='bold')
ax5.set_title('Final Epoch: Precision & Recall by Class', fontsize=14, fontweight='bold', pad=15)
ax5.set_xticks(x)
ax5.set_xticklabels(['Class 0', 'Class 1'])
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# ========== PLOT 6: TRAINING SUMMARY STATISTICS ==========
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')

# Calculate statistics
final_macro_f1 = 0.6887
final_balanced_acc = 0.6880
final_classification_report = """
Final Classification Report (Epoch 30):
────────────────────────────────────────
              precision    recall  f1-score   support

        Class 0     0.7773    0.7847    0.7810       209
        Class 1     0.6018    0.5913    0.5965       115

    accuracy                         0.7160       324
   macro avg     0.6895    0.6880    0.6887       324
weighted avg     0.7150    0.7160    0.7155       324
"""

summary_text = f"""
NEUROGATE-MRA TRAINING SUMMARY
═══════════════════════════════════════════════
• Best Evaluation Accuracy: {best_acc:.2f}% (Epoch {best_epoch})
• Final Evaluation Accuracy: {eval_acc[-1]:.2f}%
• Final Macro F1 Score: {final_macro_f1:.4f}
• Final Balanced Accuracy: {final_balanced_acc:.4f}
• Training Samples: 1109
• Evaluation Samples: 324
• Total Parameters: ~1.2M (estimated)
• Training Time: ~46s per epoch
• Best Model Saved at: Epoch 19 (72.53%)
───────────────────────────────────────────────
Key Observations:
• Model shows good convergence (loss plateau)
• Consistent improvement up to epoch 16
• Learning rate schedule improved stability
• Class 0 performs better than Class 1
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
         fontfamily='monospace', fontsize=10, va='top',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

# ========== FINAL TOUCHES ==========
fig.suptitle('NeuroGATE-MRA: EEG Gaze Classification Training Results (30 Epochs)', 
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()

# Save the figure
plt.savefig('neurogate_training_summary.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("\n" + "="*60)
print("PLOTS GENERATED SUCCESSFULLY!")
print("="*60)
print(f"✓ Accuracy progression with best: {best_acc:.1f}% at epoch {best_epoch}")
print(f"✓ Loss convergence analysis")
print(f"✓ Learning rate schedule visualization")
print(f"✓ Class imbalance analysis")
print(f"✓ Final model performance metrics")
print(f"✓ Summary statistics")
print("\nFigure saved as: 'neurogate_training_summary.png'")
print("="*60)