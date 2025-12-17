import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Extract metrics from the training log
epochs = list(range(1, 31))

# Training losses
train_loss_total = [0.7126, 0.7021, 0.6796, 0.6844, 0.6797, 0.6482, 0.6739, 0.6451, 0.6238, 0.5682,
                    0.5574, 0.5422, 0.5161, 0.4724, 0.4915, 0.4963, 0.4778, 0.4464, 0.3992, 0.4189,
                    0.4244, 0.3929, 0.4332, 0.4468, 0.4291, 0.3984, 0.3971, 0.3958, 0.4136, 0.3962]

train_loss_cls = [0.6980, 0.6895, 0.6682, 0.6738, 0.6699, 0.6386, 0.6643, 0.6359, 0.6148, 0.5589,
                  0.5472, 0.5316, 0.5048, 0.4606, 0.4788, 0.4838, 0.4656, 0.4339, 0.3866, 0.4060,
                  0.4114, 0.3798, 0.4200, 0.4334, 0.4156, 0.3847, 0.3834, 0.3819, 0.3997, 0.3823]

train_loss_gaze = [0.0291, 0.0252, 0.0228, 0.0211, 0.0196, 0.0192, 0.0191, 0.0183, 0.0181, 0.0187,
                   0.0203, 0.0212, 0.0225, 0.0234, 0.0253, 0.0251, 0.0244, 0.0250, 0.0252, 0.0259,
                   0.0260, 0.0263, 0.0264, 0.0267, 0.0271, 0.0274, 0.0273, 0.0277, 0.0278, 0.0278]

# Training accuracy
train_acc = [56.36, 55.09, 62.67, 59.15, 58.34, 61.05, 60.87, 67.90, 66.55, 73.13,
             76.19, 76.92, 79.44, 81.33, 80.07, 79.44, 83.32, 83.41, 86.20, 84.31,
             83.68, 86.38, 83.95, 82.60, 82.60, 85.21, 86.20, 84.13, 83.68, 84.85]

# Evaluation metrics
eval_loss_total = [0.6841, 0.6358, 0.6539, 0.6322, 0.6361, 0.6513, 0.5727, 0.6759, 0.5742, 0.5786,
                   0.5589, 0.5517, 0.6253, 0.5836, 0.9100, 0.5950, 0.5311, 0.5327, 0.5125, 0.5290,
                   0.5119, 0.5167, 0.5110, 0.5102, 0.5222, 0.5072, 0.5135, 0.5190, 0.5184, 0.5185]

eval_loss_cls = [0.6580, 0.6139, 0.6328, 0.6132, 0.6178, 0.6332, 0.5553, 0.6583, 0.5573, 0.5603,
                 0.5387, 0.5328, 0.6010, 0.5596, 0.8804, 0.5698, 0.5061, 0.5073, 0.4872, 0.5027,
                 0.4858, 0.4901, 0.4844, 0.4834, 0.4947, 0.4796, 0.4856, 0.4909, 0.4904, 0.4905]

eval_loss_gaze = [0.0261, 0.0219, 0.0210, 0.0190, 0.0183, 0.0181, 0.0174, 0.0177, 0.0169, 0.0183,
                  0.0203, 0.0189, 0.0243, 0.0240, 0.0296, 0.0252, 0.0250, 0.0254, 0.0253, 0.0263,
                  0.0261, 0.0267, 0.0266, 0.0268, 0.0275, 0.0277, 0.0279, 0.0281, 0.0280, 0.0280]

# Evaluation accuracy
eval_acc = [60.19, 61.73, 61.42, 62.35, 62.35, 61.73, 70.06, 63.58, 70.37, 69.75,
            71.91, 71.91, 70.68, 71.91, 62.35, 72.22, 75.31, 76.54, 78.09, 76.85,
            77.16, 77.16, 78.40, 77.78, 78.09, 78.40, 78.09, 78.40, 78.09, 78.09]

# Best model is at epoch 23
best_epoch = 23
best_eval_acc = eval_acc[best_epoch-1]
best_eval_loss = eval_loss_total[best_epoch-1]

# Learning rate schedule
lr_schedule = []
for epoch in epochs:
    if epoch <= 17:
        lr_schedule.append("2.00e-04")
    elif epoch <= 29:
        lr_schedule.append("2.00e-05")
    else:
        lr_schedule.append("2.00e-06")

# Create clean, focused figure
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Multi-Task Training Progress (30 Epochs)', fontsize=16, fontweight='bold', y=0.98)

# Colors
train_color = '#2E86AB'  # Blue
eval_color = '#A23B72'   # Purple
cls_color = '#3B8EA5'    # Teal
gaze_color = '#F18F01'   # Orange
best_color = '#2D936C'   # Green
highlight_color = '#FFD166'  # Gold

# 1. Main Accuracy Plot
ax1 = axes[0, 0]
ax1.plot(epochs, train_acc, '-', color=train_color, linewidth=2.5, marker='o', markersize=5, 
         label=f'Train (Final: {train_acc[-1]:.1f}%)', alpha=0.9)
ax1.plot(epochs, eval_acc, '-', color=eval_color, linewidth=2.5, marker='s', markersize=5, 
         label=f'Eval (Best: {best_eval_acc:.1f}%)', alpha=0.9)
ax1.axvline(x=best_epoch, color=highlight_color, linestyle='--', linewidth=2, alpha=0.8, 
           label=f'Best Epoch: {best_epoch}')
ax1.scatter(best_epoch, best_eval_acc, color=best_color, s=200, zorder=10, 
           edgecolors='black', linewidth=2, marker='*')
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Accuracy Progression', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.2, linestyle='--')
ax1.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax1.set_xticks([1, 5, 10, 15, 20, 25, 30])
ax1.set_ylim([55, 90])
# Add text annotation for best model
ax1.annotate(f'Best: {best_eval_acc:.1f}%', xy=(best_epoch, best_eval_acc), 
            xytext=(best_epoch+2, best_eval_acc+3),
            arrowprops=dict(arrowstyle='->', color=best_color, lw=1.5),
            fontsize=11, fontweight='bold', color=best_color)

# 2. Loss Progression
ax2 = axes[0, 1]
ax2.plot(epochs, train_loss_total, '-', color=train_color, linewidth=2.5, marker='o', markersize=5, 
         label=f'Train (Final: {train_loss_total[-1]:.3f})', alpha=0.9)
ax2.plot(epochs, eval_loss_total, '-', color=eval_color, linewidth=2.5, marker='s', markersize=5, 
         label=f'Eval (Best: {best_eval_loss:.3f})', alpha=0.9)
ax2.axvline(x=best_epoch, color=highlight_color, linestyle='--', linewidth=2, alpha=0.8)
ax2.scatter(best_epoch, best_eval_loss, color=best_color, s=200, zorder=10, 
           edgecolors='black', linewidth=2, marker='*')
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Total Loss', fontsize=12, fontweight='bold')
ax2.set_title('Loss Progression', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.2, linestyle='--')
ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax2.set_xticks([1, 5, 10, 15, 20, 25, 30])
ax2.set_ylim([0.4, 1.0])

# 3. Convergence Analysis
ax3 = axes[1, 0]
# Plot evaluation accuracy with moving average
window = 3
eval_acc_smooth = pd.Series(eval_acc).rolling(window=window, center=True).mean()
ax3.plot(epochs, eval_acc, 'o-', color=eval_color, linewidth=1.5, markersize=4, alpha=0.6, label='Raw')
ax3.plot(epochs, eval_acc_smooth, '-', color=eval_color, linewidth=3, alpha=0.9, 
         label=f'Moving Avg (window={window})')
ax3.axhline(y=best_eval_acc, color=best_color, linestyle='--', linewidth=2, alpha=0.7,
           label=f'Best: {best_eval_acc:.1f}%')
ax3.axvline(x=best_epoch, color=highlight_color, linestyle='--', linewidth=2, alpha=0.7)
# Highlight plateau region
plateau_start = 17
plateau_end = 30
plateau_values = eval_acc[plateau_start-1:plateau_end]
plateau_mean = np.mean(plateau_values)
plateau_std = np.std(plateau_values)
ax3.axhspan(plateau_mean - plateau_std, plateau_mean + plateau_std, alpha=0.1, color=eval_color,
           label=f'Plateau: {plateau_mean:.1f}% ± {plateau_std:.1f}%')
ax3.axvspan(plateau_start-0.5, plateau_end+0.5, alpha=0.05, color='gray')
ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax3.set_ylabel('Evaluation Accuracy (%)', fontsize=12, fontweight='bold')
ax3.set_title('Convergence Analysis', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.2, linestyle='--')
ax3.legend(loc='lower right', fontsize=9, framealpha=0.9)
ax3.set_xticks([1, 5, 10, 15, 20, 25, 30])
ax3.set_ylim([55, 85])

# 4. Performance Summary Table
ax4 = axes[1, 1]
ax4.axis('tight')
ax4.axis('off')

summary_data = [
    ['Best Epoch', f'{best_epoch}'],
    ['Best Eval Accuracy', f'{best_eval_acc:.2f}%'],
    ['Best Eval Loss', f'{best_eval_loss:.4f}'],
    ['Final Train Accuracy', f'{train_acc[-1]:.2f}%'],
    ['Final Eval Accuracy', f'{eval_acc[-1]:.2f}%'],
    ['Accuracy Gain', f'{best_eval_acc - eval_acc[0]:.2f}%'],
    ['Training Time', '~23 hours'],
    ['Learning Rate', 'Decay: 2e-4 → 2e-6'],
    ['Batch Size', 'Smaller (18 vs 35)'],
    ['Performance', 'Improved: 78.4% vs 73.46%']
]

table = ax4.table(cellText=summary_data,
                  cellLoc='left',
                  loc='center',
                  colWidths=[0.6, 0.4])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.8)

# Style the table
for i, key in enumerate(table.get_celld().keys()):
    cell = table.get_celld()[key]
    if key[0] == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#2c3e50')
    elif key[1] == 0 and key[0] in [1, 2, 3]:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#e8f4f8')
    cell.set_edgecolor('lightgray')
    cell.set_linewidth(0.5)

ax4.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()

# Create a second figure for detailed analysis
fig2, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2, figsize=(15, 10))
fig2.suptitle('Training Dynamics Analysis', fontsize=16, fontweight='bold', y=0.98)

# 5. Train-Eval Gap Analysis
train_eval_gap = [train - eval for train, eval in zip(train_acc, eval_acc)]
ax5.bar(epochs, train_eval_gap, color=train_color, alpha=0.6, edgecolor=train_color, linewidth=0.5)
ax5.axhline(y=np.mean(train_eval_gap), color='red', linestyle='--', linewidth=2, 
           label=f'Mean Gap: {np.mean(train_eval_gap):.2f}%')
ax5.axvline(x=best_epoch, color=highlight_color, linestyle=':', linewidth=1.5, alpha=0.6)
ax5.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax5.set_ylabel('Train-Eval Gap (%)', fontsize=12, fontweight='bold')
ax5.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.2, linestyle='--', axis='y')
ax5.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax5.set_xticks([1, 5, 10, 15, 20, 25, 30])
ax5.set_ylim([-5, 25])

# 6. Loss Components Analysis
ax6.plot(epochs, train_loss_cls, '-', color=cls_color, linewidth=2, label='Train CLS Loss', alpha=0.8)
ax6.plot(epochs, eval_loss_cls, '--', color=cls_color, linewidth=1.5, label='Eval CLS Loss', alpha=0.6)
ax6.plot(epochs, train_loss_gaze, '-', color=gaze_color, linewidth=2, label='Train Gaze Loss', alpha=0.8)
ax6.plot(epochs, eval_loss_gaze, '--', color=gaze_color, linewidth=1.5, label='Eval Gaze Loss', alpha=0.6)
ax6.axvline(x=best_epoch, color=highlight_color, linestyle=':', linewidth=1.5, alpha=0.6)
ax6.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax6.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax6.set_title('Loss Components', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.2, linestyle='--')
ax6.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax6.set_xticks([1, 5, 10, 15, 20, 25, 30])
ax6.set_ylim([0, 0.9])

# 7. Learning Rate Effects
# Create phases based on LR changes
phases = [(1, 17, 'High LR\n(2e-4)', '#FF9999'),
          (18, 29, 'Low LR\n(2e-5)', '#99FF99'),
          (30, 30, 'Very Low\n(2e-6)', '#9999FF')]

for start, end, label, color in phases:
    ax7.axvspan(start-0.5, end+0.5, alpha=0.2, color=color)
    if end-start > 1:  # Only label phases with significant duration
        ax7.text((start+end)/2, 58, label, ha='center', va='center', 
                fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor='white', alpha=0.8, edgecolor='gray'))

ax7.plot(epochs, eval_acc, '-', color=eval_color, linewidth=3, marker='s', markersize=6)
ax7.axvline(x=best_epoch, color=highlight_color, linestyle='--', linewidth=2, alpha=0.7)
ax7.scatter(best_epoch, best_eval_acc, color=best_color, s=200, zorder=10, 
           edgecolors='black', linewidth=2, marker='*')
ax7.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax7.set_ylabel('Evaluation Accuracy (%)', fontsize=12, fontweight='bold')
ax7.set_title('Learning Rate Effects', fontsize=14, fontweight='bold')
ax7.grid(True, alpha=0.2, linestyle='--')
ax7.set_xticks([1, 5, 10, 15, 20, 25, 30])
ax7.set_ylim([55, 85])

# 8. Performance Comparison with Previous Runs
ax8.axis('tight')
ax8.axis('off')

comparison_data = [
    ['Run', 'Best Acc', 'Epochs', 'Batch Size', 'LR Schedule'],
    ['Previous (40)', '73.46%', '40', '35', '5e-4 → 5e-9'],
    ['Current (30)', '78.40%', '30', '18', '2e-4 → 2e-6'],
    ['Improvement', '+4.94%', '-10', '-17', 'Simpler']
]

table2 = ax8.table(cellText=comparison_data,
                   cellLoc='center',
                   loc='center',
                   colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])

table2.auto_set_font_size(False)
table2.set_fontsize(10)
table2.scale(1, 2)

# Style the table
for i, key in enumerate(table2.get_celld().keys()):
    cell = table2.get_celld()[key]
    if key[0] == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#2c3e50')
    elif key[1] == 0 and key[0] in [2, 3]:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#e8f4f8')
    cell.set_edgecolor('lightgray')
    cell.set_linewidth(0.5)

ax8.set_title('Comparison with Previous Run', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()

# Print concise analysis
print("="*80)
print("TRAINING SUMMARY")
print("="*80)
print(f"\n🏆 BEST PERFORMANCE:")
print(f"   • Epoch: {best_epoch}")
print(f"   • Evaluation Accuracy: {best_eval_acc:.2f}%")
print(f"   • Evaluation Loss: {best_eval_loss:.4f}")
print(f"   • Improvement from start: {best_eval_acc - eval_acc[0]:.2f}%")
print(f"   • Improvement from previous run: +4.94%")

print(f"\n📊 FINAL PERFORMANCE:")
print(f"   • Training Accuracy: {train_acc[-1]:.2f}%")
print(f"   • Evaluation Accuracy: {eval_acc[-1]:.2f}%")
print(f"   • Train-Eval Gap: {train_acc[-1] - eval_acc[-1]:.2f}%")
print(f"   • Avg Train-Eval Gap: {np.mean(train_eval_gap):.2f}%")

print(f"\n🔍 KEY OBSERVATIONS:")
print(f"   • Best performance achieved at epoch {best_epoch} (after LR reduction)")
print(f"   • Model plateaus around 77-78% accuracy after epoch 17")
print(f"   • Smaller batch size (18 vs 35) showed better performance")
print(f"   • Learning rate reduction at epoch 17 was effective")
print(f"   • Gaze loss remains stable throughout training")

print(f"\n📈 COMPARISON WITH PREVIOUS RUN:")
print(f"   • Accuracy: 78.40% vs 73.46% (+4.94% improvement)")
print(f"   • Epochs: 30 vs 40 (more efficient training)")
print(f"   • Batch Size: 18 vs 35 (smaller batches helped)")
print(f"   • LR Schedule: Simpler decay pattern")

print(f"\n💡 RECOMMENDATIONS:")
print(f"   • Best model saved at epoch {best_epoch}")
print(f"   • Consider early stopping around epoch 25")
print(f"   • Current setup is working well - continue with similar parameters")
print(f"   • Could try even smaller learning rate for final fine-tuning")
print("="*80)