import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Extract metrics from the training log
epochs = list(range(1, 41))

# Training losses (total, classification, gaze)
train_loss_total = [0.7212, 0.7001, 0.6672, 0.6673, 0.6683, 0.6359, 0.6167, 0.5597, 0.5236, 0.5223,
                    0.5071, 0.5122, 0.4578, 0.5138, 0.5084, 0.4535, 0.4651, 0.4792, 0.4793, 0.5093,
                    0.4996, 0.4788, 0.4900, 0.4649, 0.4569, 0.4786, 0.4454, 0.4833, 0.4438, 0.4660,
                    0.4540, 0.4497, 0.5024, 0.4719, 0.4669, 0.4992, 0.4836, 0.4620, 0.4723, 0.4577]

train_loss_cls = [0.7013, 0.6867, 0.6546, 0.6544, 0.6563, 0.6235, 0.6027, 0.5427, 0.5045, 0.5027,
                  0.4870, 0.4918, 0.4367, 0.4920, 0.4869, 0.4312, 0.4421, 0.4561, 0.4564, 0.4864,
                  0.4767, 0.4558, 0.4672, 0.4420, 0.4339, 0.4557, 0.4224, 0.4604, 0.4207, 0.4432,
                  0.4310, 0.4267, 0.4795, 0.4488, 0.4440, 0.4762, 0.4607, 0.4389, 0.4493, 0.4347]

train_loss_gaze = [0.0265, 0.0178, 0.0168, 0.0172, 0.0160, 0.0166, 0.0187, 0.0227, 0.0254, 0.0260,
                   0.0268, 0.0272, 0.0281, 0.0291, 0.0286, 0.0298, 0.0307, 0.0308, 0.0306, 0.0304,
                   0.0305, 0.0307, 0.0304, 0.0306, 0.0307, 0.0306, 0.0307, 0.0305, 0.0307, 0.0304,
                   0.0307, 0.0306, 0.0305, 0.0307, 0.0305, 0.0306, 0.0306, 0.0307, 0.0306, 0.0306]

# Training accuracy
train_acc = [52.93, 54.01, 61.68, 62.49, 63.21, 67.00, 69.61, 76.19, 77.82, 78.90,
             78.99, 79.35, 82.69, 79.62, 79.89, 83.59, 83.59, 83.50, 82.78, 80.70,
             81.15, 83.86, 82.42, 83.68, 83.95, 82.24, 84.67, 82.42, 83.05, 83.32,
             84.94, 83.86, 81.24, 82.87, 83.41, 82.60, 83.05, 83.50, 83.14, 82.24]

# Evaluation metrics
eval_loss_total = [0.7257, 0.6849, 0.6184, 0.6140, 0.6536, 0.6305, 0.6091, 0.5552, 0.6012, 0.6327,
                   0.5744, 0.6081, 0.5827, 0.5820, 0.6371, 0.5983, 0.5735, 0.5751, 0.5698, 0.5841,
                   0.5960, 0.5776, 0.5712, 0.5964, 0.5816, 0.5890, 0.5822, 0.5857, 0.5947, 0.5769,
                   0.5712, 0.5848, 0.5798, 0.5860, 0.5798, 0.5767, 0.5794, 0.5795, 0.5954, 0.5942]

eval_loss_cls = [0.7042, 0.6685, 0.5992, 0.5987, 0.6383, 0.6138, 0.5877, 0.5315, 0.5756, 0.6056,
                 0.5470, 0.5804, 0.5537, 0.5536, 0.6073, 0.5679, 0.5429, 0.5448, 0.5390, 0.5532,
                 0.5649, 0.5469, 0.5404, 0.5654, 0.5508, 0.5580, 0.5515, 0.5550, 0.5635, 0.5466,
                 0.5405, 0.5541, 0.5491, 0.5554, 0.5484, 0.5460, 0.5491, 0.5488, 0.5638, 0.5631]

eval_loss_gaze = [0.0215, 0.0163, 0.0192, 0.0153, 0.0153, 0.0166, 0.0214, 0.0238, 0.0256, 0.0271,
                  0.0274, 0.0277, 0.0291, 0.0284, 0.0298, 0.0304, 0.0306, 0.0303, 0.0308, 0.0310,
                  0.0311, 0.0306, 0.0308, 0.0310, 0.0309, 0.0310, 0.0307, 0.0308, 0.0312, 0.0303,
                  0.0307, 0.0308, 0.0307, 0.0306, 0.0314, 0.0307, 0.0303, 0.0307, 0.0316, 0.0310]

# Evaluation accuracy
eval_acc = [55.86, 62.96, 65.12, 67.59, 63.89, 66.05, 66.98, 73.46, 70.99, 68.83,
            71.91, 70.06, 71.30, 71.91, 69.75, 70.68, 72.22, 71.91, 71.30, 71.91,
            72.53, 71.60, 71.30, 71.91, 71.60, 72.22, 71.91, 72.53, 72.53, 71.30,
            71.30, 72.22, 71.60, 72.22, 71.91, 71.60, 71.91, 71.91, 72.84, 72.22]

# Best model is at epoch 8
best_epoch = 8
best_eval_acc = eval_acc[best_epoch-1]
best_eval_loss = eval_loss_total[best_epoch-1]

# Learning rate schedule based on epochs
lr_schedule = []
for epoch in epochs:
    if epoch <= 8:
        lr_schedule.append("5.00e-04")
    elif epoch <= 18:
        lr_schedule.append("5.00e-05")
    elif epoch <= 21:
        lr_schedule.append("5.00e-06")
    elif epoch <= 27:
        lr_schedule.append("5.00e-07")
    elif epoch <= 28:
        lr_schedule.append("5.00e-08")
    else:
        lr_schedule.append("5.00e-09")

# Create a clean, focused figure
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Multi-Task Training Progress (40 Epochs)', fontsize=16, fontweight='bold', y=0.98)

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
ax1.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40])
ax1.set_ylim([50, 85])
# Add text annotation for best model
ax1.annotate(f'Best: {best_eval_acc:.1f}%', xy=(best_epoch, best_eval_acc), 
            xytext=(best_epoch+3, best_eval_acc+2),
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
ax2.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40])
ax2.set_ylim([0.4, 0.8])

# 3. Loss Components Comparison
ax3 = axes[1, 0]
x = np.arange(len(epochs))
width = 0.35
# Calculate ratios
train_cls_ratio = [cls/total for cls, total in zip(train_loss_cls, train_loss_total)]
train_gaze_ratio = [gaze/total for gaze, total in zip(train_loss_gaze, train_loss_total)]
eval_cls_ratio = [cls/total for cls, total in zip(eval_loss_cls, eval_loss_total)]
eval_gaze_ratio = [gaze/total for gaze, total in zip(eval_loss_gaze, eval_loss_total)]

# Plot as area chart for clarity
ax3.stackplot(epochs, train_cls_ratio, train_gaze_ratio, 
              colors=[cls_color, gaze_color], alpha=0.7, labels=['CLS Loss', 'Gaze Loss'])
ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax3.set_ylabel('Loss Ratio', fontsize=12, fontweight='bold')
ax3.set_title('Training Loss Composition', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.2, linestyle='--')
ax3.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax3.set_xticks([1, 10, 20, 30, 40])
ax3.set_ylim([0, 1])

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
    ['Final Train Loss', f'{train_loss_total[-1]:.4f}'],
    ['Final Eval Loss', f'{eval_loss_total[-1]:.4f}'],
    ['Accuracy Gain', f'{best_eval_acc - eval_acc[0]:.2f}%'],
    ['Training Time', '~28 hours'],
    ['Learning Rate', 'Decay: 5e-4 → 5e-9']
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

# 5. Classification vs Gaze Loss
ax5.plot(epochs, train_loss_cls, '-', color=cls_color, linewidth=2, label='Train CLS Loss', alpha=0.8)
ax5.plot(epochs, eval_loss_cls, '--', color=cls_color, linewidth=2, label='Eval CLS Loss', alpha=0.6)
ax5.plot(epochs, train_loss_gaze, '-', color=gaze_color, linewidth=2, label='Train Gaze Loss', alpha=0.8)
ax5.plot(epochs, eval_loss_gaze, '--', color=gaze_color, linewidth=2, label='Eval Gaze Loss', alpha=0.6)
ax5.axvline(x=best_epoch, color=highlight_color, linestyle=':', linewidth=1.5, alpha=0.6)
ax5.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax5.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax5.set_title('Classification vs Gaze Loss', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.2, linestyle='--')
ax5.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax5.set_xticks([1, 10, 20, 30, 40])
ax5.set_ylim([0, 0.8])

# 6. Train-Eval Gap Analysis
train_eval_gap = [train - eval for train, eval in zip(train_acc, eval_acc)]
ax6.bar(epochs, train_eval_gap, color=train_color, alpha=0.6, edgecolor=train_color, linewidth=0.5)
ax6.axhline(y=np.mean(train_eval_gap), color='red', linestyle='--', linewidth=2, 
           label=f'Mean Gap: {np.mean(train_eval_gap):.2f}%')
ax6.axvline(x=best_epoch, color=highlight_color, linestyle=':', linewidth=1.5, alpha=0.6)
ax6.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax6.set_ylabel('Train-Eval Gap (%)', fontsize=12, fontweight='bold')
ax6.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.2, linestyle='--', axis='y')
ax6.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax6.set_xticks([1, 10, 20, 30, 40])
ax6.set_ylim([-10, 20])

# 7. Convergence Analysis (Zoom on evaluation accuracy)
ax7.plot(epochs, eval_acc, '-', color=eval_color, linewidth=3, marker='s', markersize=6, alpha=0.9)
ax7.axhline(y=best_eval_acc, color=best_color, linestyle='--', linewidth=2, alpha=0.7,
           label=f'Best: {best_eval_acc:.1f}%')
ax7.axvline(x=best_epoch, color=highlight_color, linestyle='--', linewidth=2, alpha=0.7)
ax7.fill_between(epochs, best_eval_acc-1, best_eval_acc+1, alpha=0.1, color=best_color,
                label='±1% margin')
ax7.scatter(best_epoch, best_eval_acc, color=best_color, s=250, zorder=10, 
           edgecolors='black', linewidth=2, marker='*')
ax7.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax7.set_ylabel('Evaluation Accuracy (%)', fontsize=12, fontweight='bold')
ax7.set_title('Convergence Analysis', fontsize=14, fontweight='bold')
ax7.grid(True, alpha=0.2, linestyle='--')
ax7.legend(loc='lower right', fontsize=9, framealpha=0.9)
ax7.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40])
ax7.set_ylim([55, 80])

# 8. Learning Rate Effects
# Create phases based on LR changes
phases = [(1, 8, 'High LR\n(5e-4)', '#FF9999'),
          (9, 18, 'Med LR\n(5e-5)', '#99FF99'),
          (19, 21, 'Low LR\n(5e-6)', '#9999FF'),
          (22, 27, 'Very Low\n(5e-7)', '#FFCC99'),
          (28, 28, 'Ultra Low\n(5e-8)', '#CC99FF'),
          (29, 40, 'Minimal\n(5e-9)', '#99FFFF')]

for start, end, label, color in phases:
    ax8.axvspan(start-0.5, end+0.5, alpha=0.2, color=color)
    if end-start > 2:  # Only label phases with significant duration
        ax8.text((start+end)/2, 56, label, ha='center', va='center', 
                fontsize=8, fontweight='bold')

ax8.plot(epochs, eval_acc, '-', color=eval_color, linewidth=3, marker='s', markersize=6)
ax8.axvline(x=best_epoch, color=highlight_color, linestyle='--', linewidth=2, alpha=0.7)
ax8.scatter(best_epoch, best_eval_acc, color=best_color, s=200, zorder=10, 
           edgecolors='black', linewidth=2, marker='*')
ax8.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax8.set_ylabel('Evaluation Accuracy (%)', fontsize=12, fontweight='bold')
ax8.set_title('Learning Rate Schedule Effects', fontsize=14, fontweight='bold')
ax8.grid(True, alpha=0.2, linestyle='--')
ax8.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40])
ax8.set_ylim([55, 80])

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()

# Print concise analysis
print("="*80)
print("TRAINING SUMMARY")
print("="*80)
print(f"\n📊 BEST PERFORMANCE:")
print(f"   • Epoch: {best_epoch}")
print(f"   • Evaluation Accuracy: {best_eval_acc:.2f}%")
print(f"   • Evaluation Loss: {best_eval_loss:.4f}")
print(f"   • Improvement from start: {best_eval_acc - eval_acc[0]:.2f}%")

print(f"\n📈 FINAL PERFORMANCE:")
print(f"   • Training Accuracy: {train_acc[-1]:.2f}%")
print(f"   • Evaluation Accuracy: {eval_acc[-1]:.2f}%")
print(f"   • Train-Eval Gap: {train_acc[-1] - eval_acc[-1]:.2f}%")

print(f"\n🔍 KEY OBSERVATIONS:")
print(f"   • Best performance achieved early (epoch {best_epoch})")
print(f"   • Model plateaus around 71-73% accuracy after epoch 8")
print(f"   • Gaze loss stabilizes early and remains low")
print(f"   • Moderate overfitting (avg train-eval gap: {np.mean(train_eval_gap):.2f}%)")
print(f"   • Learning rate decay effective in maintaining performance")

print(f"\n💡 RECOMMENDATIONS:")
print(f"   • Consider early stopping around epoch 10-15")
print(f"   • Best model saved at epoch {best_epoch}")
print(f"   • For next run: Try different learning rate schedule")
print(f"   • Could benefit from regularization to reduce overfitting")
print("="*80)