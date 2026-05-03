import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

# Extract metrics from the training log
epochs = list(range(1, 41))

# Training losses (total, classification, gaze)
train_loss_total = [0.7185, 0.6987, 0.6801, 0.6703, 0.6308, 0.5610, 0.5411, 0.4982, 
                   0.5525, 0.5346, 0.4643, 0.4278, 0.4359, 0.4277, 0.4298, 0.4301,
                   0.4472, 0.4217, 0.4593, 0.3888, 0.4624, 0.4042, 0.4376, 0.4491,
                   0.4123, 0.4145, 0.4262, 0.4418, 0.3990, 0.4138, 0.4347, 0.4372,
                   0.3999, 0.4221, 0.4233, 0.3964, 0.4318, 0.4549, 0.4303, 0.4060]

train_loss_cls = [0.7000, 0.6858, 0.6692, 0.6601, 0.6207, 0.5495, 0.5272, 0.4845,
                  0.5368, 0.5180, 0.4487, 0.4120, 0.4196, 0.4110, 0.4130, 0.4132,
                  0.4302, 0.4046, 0.4419, 0.3717, 0.4451, 0.3868, 0.4203, 0.4317,
                  0.3950, 0.3971, 0.4088, 0.4244, 0.3816, 0.3965, 0.4175, 0.4198,
                  0.3826, 0.4048, 0.4059, 0.3791, 0.4144, 0.4375, 0.4130, 0.3886]

train_loss_gaze = [0.0308, 0.0215, 0.0182, 0.0171, 0.0169, 0.0192, 0.0232, 0.0228,
                   0.0244, 0.0244, 0.0260, 0.0264, 0.0271, 0.0278, 0.0279, 0.0282,
                   0.0284, 0.0286, 0.0289, 0.0286, 0.0288, 0.0289, 0.0289, 0.0289,
                   0.0288, 0.0290, 0.0289, 0.0290, 0.0289, 0.0288, 0.0288, 0.0290,
                   0.0288, 0.0289, 0.0290, 0.0289, 0.0291, 0.0289, 0.0288, 0.0289]

# Training accuracy
train_acc = [54.73, 52.93, 60.05, 56.72, 63.93, 74.66, 77.73, 77.46, 77.82, 78.72,
             83.14, 84.49, 82.78, 84.22, 83.95, 83.50, 82.69, 85.48, 82.69, 86.65,
             82.69, 86.20, 83.05, 83.23, 85.66, 85.12, 84.22, 84.13, 86.47, 85.48,
             83.68, 83.32, 85.39, 83.32, 84.76, 86.20, 84.22, 81.79, 84.13, 85.93]

# Evaluation metrics
eval_loss_total = [0.6487, 0.6713, 0.6643, 0.6708, 0.5962, 0.5750, 0.6386, 0.6279,
                   0.6735, 0.6455, 0.5681, 0.5745, 0.5612, 0.5605, 0.5650, 0.5425,
                   0.5567, 0.5730, 0.5618, 0.5514, 0.5544, 0.5440, 0.5479, 0.5511,
                   0.5440, 0.5476, 0.5483, 0.5473, 0.5455, 0.5432, 0.5422, 0.5443,
                   0.5459, 0.5452, 0.5455, 0.5416, 0.5423, 0.5443, 0.5459, 0.5440]

eval_loss_cls = [0.6241, 0.6525, 0.6479, 0.6548, 0.5793, 0.5533, 0.6161, 0.6030,
                 0.6476, 0.6189, 0.5412, 0.5468, 0.5329, 0.5314, 0.5357, 0.5134,
                 0.5274, 0.5425, 0.5318, 0.5219, 0.5242, 0.5144, 0.5180, 0.5209,
                 0.5145, 0.5175, 0.5183, 0.5173, 0.5156, 0.5135, 0.5127, 0.5145,
                 0.5163, 0.5155, 0.5155, 0.5120, 0.5126, 0.5144, 0.5159, 0.5144]

eval_loss_gaze = [0.0246, 0.0189, 0.0164, 0.0160, 0.0169, 0.0217, 0.0224, 0.0249,
                  0.0259, 0.0265, 0.0269, 0.0277, 0.0283, 0.0291, 0.0293, 0.0290,
                  0.0293, 0.0305, 0.0301, 0.0295, 0.0302, 0.0296, 0.0299, 0.0302,
                  0.0296, 0.0301, 0.0301, 0.0300, 0.0298, 0.0297, 0.0295, 0.0298,
                  0.0296, 0.0298, 0.0300, 0.0296, 0.0298, 0.0299, 0.0300, 0.0296]

# Evaluation accuracy
eval_acc = [63.89, 61.11, 62.96, 62.35, 71.30, 73.77, 68.52, 67.59, 67.90, 69.44,
            74.38, 73.15, 75.62, 75.00, 75.31, 76.23, 75.62, 74.07, 75.31, 77.47,
            76.85, 77.47, 76.54, 76.23, 77.47, 76.23, 76.54, 75.93, 77.16, 76.54,
            76.54, 77.16, 76.54, 77.47, 76.54, 76.54, 76.85, 76.54, 76.54, 77.16]

# Best model is at epoch 20
best_epoch = 20
best_eval_acc = eval_acc[best_epoch-1]
best_eval_loss = eval_loss_total[best_epoch-1]

# Learning rate schedule based on epochs
lr_schedule = []
lr_values = []
for epoch in epochs:
    if epoch <= 10:
        lr_schedule.append("3.00e-04")
        lr_values.append(3e-4)
    elif epoch <= 19:
        lr_schedule.append("3.00e-05")
        lr_values.append(3e-5)
    elif epoch <= 24:
        lr_schedule.append("3.00e-06")
        lr_values.append(3e-6)
    elif epoch <= 27:
        lr_schedule.append("3.00e-07")
        lr_values.append(3e-7)
    elif epoch <= 33:
        lr_schedule.append("3.00e-08")
        lr_values.append(3e-8)
    else:
        lr_schedule.append("3.00e-09")
        lr_values.append(3e-9)

# Create figure with subplots
fig = plt.figure(figsize=(20, 16))
fig.suptitle('Multi-Task Training Progress Analysis (40 Epochs)', fontsize=20, fontweight='bold', y=0.98)

# Define grid
gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

# Colors
train_color = '#1f77b4'  # Blue
eval_color = '#ff7f0e'   # Orange
cls_color = '#2ca02c'    # Green
gaze_color = '#d62728'   # Red
best_color = '#9467bd'   # Purple
highlight_color = '#ffd700'  # Gold for highlighting
lr_color = '#8c564b'     # Brown

# 1. Total Loss curves
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(epochs, train_loss_total, 'o-', color=train_color, linewidth=2, markersize=5, label='Train Total Loss', alpha=0.8)
ax1.plot(epochs, eval_loss_total, 's-', color=eval_color, linewidth=2, markersize=5, label='Eval Total Loss', alpha=0.8)
ax1.axvline(x=best_epoch, color=highlight_color, linestyle='--', linewidth=2, alpha=0.7, label=f'Best Model (Epoch {best_epoch})')
ax1.scatter(best_epoch, best_eval_loss, color=best_color, s=150, zorder=5, edgecolors='black', linewidth=2, marker='*', label=f'Best Eval Loss: {best_eval_loss:.4f}')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Total Loss', fontsize=12)
ax1.set_title('Total Loss Progression', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', fontsize=10)
ax1.set_xticks(np.arange(1, 41, 4))

# 2. Accuracy curves
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(epochs, train_acc, 'o-', color=train_color, linewidth=2, markersize=5, label='Train Acc', alpha=0.8)
ax2.plot(epochs, eval_acc, 's-', color=eval_color, linewidth=2, markersize=5, label='Eval Acc', alpha=0.8)
ax2.axvline(x=best_epoch, color=highlight_color, linestyle='--', linewidth=2, alpha=0.7, label=f'Best Model (Epoch {best_epoch})')
ax2.scatter(best_epoch, best_eval_acc, color=best_color, s=150, zorder=5, edgecolors='black', linewidth=2, marker='*', label=f'Best Eval Acc: {best_eval_acc:.2f}%')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Accuracy Progression', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='lower right', fontsize=10)
ax2.set_xticks(np.arange(1, 41, 4))

# 3. Classification Loss components
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(epochs, train_loss_cls, 'o-', color=cls_color, linewidth=2, markersize=5, label='Train CLS Loss', alpha=0.8)
ax3.plot(epochs, eval_loss_cls, 's-', color=cls_color, linewidth=1.5, markersize=4, label='Eval CLS Loss', alpha=0.6)
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Classification Loss', fontsize=12)
ax3.set_title('Classification Loss Components', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper right', fontsize=10)
ax3.set_xticks(np.arange(1, 41, 4))

# 4. Gaze Loss components
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(epochs, train_loss_gaze, 'o-', color=gaze_color, linewidth=2, markersize=5, label='Train Gaze Loss', alpha=0.8)
ax4.plot(epochs, eval_loss_gaze, 's-', color=gaze_color, linewidth=1.5, markersize=4, label='Eval Gaze Loss', alpha=0.6)
ax4.set_xlabel('Epoch', fontsize=12)
ax4.set_ylabel('Gaze Loss', fontsize=12)
ax4.set_title('Gaze Loss Components', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(loc='upper right', fontsize=10)
ax4.set_xticks(np.arange(1, 41, 4))

# 5. Learning rate schedule
ax5 = fig.add_subplot(gs[1, 1])
ax5.semilogy(epochs, lr_values, 'o-', color=lr_color, linewidth=2, markersize=5)
ax5.axvline(x=best_epoch, color=highlight_color, linestyle='--', linewidth=2, alpha=0.7)
ax5.set_xlabel('Epoch', fontsize=12)
ax5.set_ylabel('Learning Rate (log scale)', fontsize=12)
ax5.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3, which='both')
ax5.set_xticks(np.arange(1, 41, 4))
# Add annotations for LR changes
for change_point in [10, 19, 24, 27, 33]:
    ax5.axvline(x=change_point, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax5.text(change_point+0.5, 1e-8, f'LR↓', fontsize=9, rotation=90, verticalalignment='bottom')

# 6. Zoomed accuracy (last 20 epochs)
ax6 = fig.add_subplot(gs[1, 2])
zoom_start = 20
ax6.plot(epochs[zoom_start-1:], train_acc[zoom_start-1:], 'o-', color=train_color, linewidth=2, markersize=5, label='Train Acc')
ax6.plot(epochs[zoom_start-1:], eval_acc[zoom_start-1:], 's-', color=eval_color, linewidth=2, markersize=5, label='Eval Acc')
ax6.axvline(x=best_epoch, color=highlight_color, linestyle='--', linewidth=2, alpha=0.7, label=f'Best Model (Epoch {best_epoch})')
ax6.scatter(best_epoch, best_eval_acc, color=best_color, s=150, zorder=5, edgecolors='black', linewidth=2, marker='*')
ax6.set_xlabel('Epoch', fontsize=12)
ax6.set_ylabel('Accuracy (%)', fontsize=12)
ax6.set_title('Zoom: Accuracy (Epochs 20-40)', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.legend(loc='lower right', fontsize=10)
ax6.set_ylim([70, 90])
ax6.set_xticks(np.arange(zoom_start, 41, 2))

# 7. Loss breakdown comparison
ax7 = fig.add_subplot(gs[2, 0:2])
bar_width = 0.35
x = np.arange(len(epochs))
train_cls_ratio = [cls/total for cls, total in zip(train_loss_cls, train_loss_total)]
train_gaze_ratio = [gaze/total for gaze, total in zip(train_loss_gaze, train_loss_total)]
ax7.bar(x - bar_width/2, train_cls_ratio, bar_width, label='Train CLS Ratio', color=cls_color, alpha=0.7)
ax7.bar(x - bar_width/2, train_gaze_ratio, bar_width, bottom=train_cls_ratio, label='Train Gaze Ratio', color=gaze_color, alpha=0.7)
eval_cls_ratio = [cls/total for cls, total in zip(eval_loss_cls, eval_loss_total)]
eval_gaze_ratio = [gaze/total for gaze, total in zip(eval_loss_gaze, eval_loss_total)]
ax7.bar(x + bar_width/2, eval_cls_ratio, bar_width, label='Eval CLS Ratio', color=cls_color, alpha=0.4, hatch='//')
ax7.bar(x + bar_width/2, eval_gaze_ratio, bar_width, bottom=eval_cls_ratio, label='Eval Gaze Ratio', color=gaze_color, alpha=0.4, hatch='//')
ax7.set_xlabel('Epoch', fontsize=12)
ax7.set_ylabel('Loss Ratio', fontsize=12)
ax7.set_title('Loss Component Ratios (Stacked)', fontsize=14, fontweight='bold')
ax7.set_xticks(x[::4])
ax7.set_xticklabels(epochs[::4])
ax7.legend(loc='upper right', fontsize=9, ncol=2)
ax7.grid(True, alpha=0.3, axis='y')

# 8. Performance summary table
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis('tight')
ax8.axis('off')
summary_data = {
    'Metric': ['Best Epoch', 'Best Evaluation Accuracy', 'Best Evaluation Loss',
               'Final Training Accuracy', 'Final Evaluation Accuracy',
               'Final Training Loss', 'Final Evaluation Loss',
               'Accuracy Improvement (Epoch 1 to Best)',
               'Training Time', 'Learning Rate Schedule',
               'Multi-Task: CLS Loss Weight', 'Multi-Task: Gaze Loss Weight'],
    'Value': [f'{best_epoch}', f'{best_eval_acc:.2f}%', f'{best_eval_loss:.4f}',
              f'{train_acc[-1]:.2f}%', f'{eval_acc[-1]:.2f}%',
              f'{train_loss_total[-1]:.4f}', f'{eval_loss_total[-1]:.4f}',
              f'{(best_eval_acc - eval_acc[0]):.2f}%',
              '~30 hours', 'Stepwise decay',
              'Main (varies)', '~0.02-0.03']
}
table = ax8.table(cellText=list(zip(summary_data['Metric'], summary_data['Value'])),
                  colLabels=['Metric', 'Value'],
                  cellLoc='left',
                  loc='center',
                  colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.8)
for i, key in enumerate(table.get_celld().keys()):
    cell = table.get_celld()[key]
    if key[0] == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#2c3e50')
    elif key[1] == 0 and key[0] in [1, 2, 3]:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#e8f4f8')
ax8.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)

# 9. Training dynamics analysis
ax9 = fig.add_subplot(gs[3, :])
# Create phases based on LR changes
phases = [(1, 10, 'High LR (3e-4)', '#FFDDC1'),
          (11, 19, 'Medium LR (3e-5)', '#FFABAB'),
          (20, 24, 'Low LR (3e-6)', '#FFC3A0'),
          (25, 27, 'Very Low LR (3e-7)', '#D5AAFF'),
          (28, 33, 'Ultra Low LR (3e-8)', '#85E3FF'),
          (34, 40, 'Minimal LR (3e-9)', '#AFF8D8')]

# Plot accuracy with phases
for start, end, label, color in phases:
    ax9.axvspan(start-0.5, end+0.5, alpha=0.2, color=color)
    ax9.text((start+end)/2, 50, label, ha='center', va='center', fontsize=9, rotation=90)

ax9.plot(epochs, train_acc, 'o-', color=train_color, linewidth=2, markersize=4, label='Train Accuracy')
ax9.plot(epochs, eval_acc, 's-', color=eval_color, linewidth=2, markersize=4, label='Evaluation Accuracy')
ax9.axvline(x=best_epoch, color=highlight_color, linestyle='--', linewidth=3, alpha=0.8, label=f'Best Model: Epoch {best_epoch} ({best_eval_acc}%)')
ax9.fill_between(epochs, train_acc, eval_acc, alpha=0.2, color='gray', label='Train-Eval Gap')
ax9.set_xlabel('Epoch', fontsize=12)
ax9.set_ylabel('Accuracy (%)', fontsize=12)
ax9.set_title('Training Phases with Learning Rate Decay', fontsize=14, fontweight='bold')
ax9.grid(True, alpha=0.3)
ax9.legend(loc='lower right', fontsize=10)
ax9.set_ylim([50, 90])
ax9.set_xticks(np.arange(1, 41, 2))

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()

# Create a separate figure for combined view
fig2, ((ax10, ax11), (ax12, ax13)) = plt.subplots(2, 2, figsize=(18, 12))
fig2.suptitle('Multi-Task Training: Key Insights', fontsize=18, fontweight='bold', y=0.98)

# Combined loss plot with components
ax10.plot(epochs, train_loss_total, 'o-', color=train_color, linewidth=2, markersize=4, label='Train Total', alpha=0.8)
ax10.plot(epochs, eval_loss_total, 's-', color=eval_color, linewidth=2, markersize=4, label='Eval Total', alpha=0.8)
ax10.plot(epochs, train_loss_cls, ':', color=cls_color, linewidth=1.5, label='Train CLS', alpha=0.6)
ax10.plot(epochs, eval_loss_cls, ':', color=cls_color, linewidth=1, label='Eval CLS', alpha=0.4)
ax10.axvline(x=best_epoch, color=highlight_color, linestyle='--', linewidth=2.5, alpha=0.9, label=f'Best Model (Epoch {best_epoch})')
ax10.scatter(best_epoch, best_eval_loss, color=best_color, s=200, zorder=10, 
           edgecolors='black', linewidth=2, marker='*', label=f'Best Eval Loss: {best_eval_loss:.4f}')
ax10.axvspan(best_epoch-0.5, best_epoch+0.5, alpha=0.2, color=highlight_color)
ax10.set_xlabel('Epoch', fontsize=12)
ax10.set_ylabel('Loss', fontsize=12)
ax10.set_title('Loss Components with Best Model Highlighted', fontsize=14, fontweight='bold')
ax10.grid(True, alpha=0.3)
ax10.legend(loc='upper right', fontsize=9)
ax10.set_xticks(np.arange(1, 41, 4))

# Combined accuracy plot with moving average
ax11.plot(epochs, train_acc, 'o-', color=train_color, linewidth=2, markersize=4, label='Train Accuracy', alpha=0.8)
ax11.plot(epochs, eval_acc, 's-', color=eval_color, linewidth=2, markersize=4, label='Evaluation Accuracy', alpha=0.8)
# Add moving average for evaluation accuracy
window_size = 3
eval_acc_ma = pd.Series(eval_acc).rolling(window=window_size, center=True).mean()
ax11.plot(epochs, eval_acc_ma, '-', color='black', linewidth=2, label=f'Eval MA ({window_size}-epoch)', alpha=0.6)
ax11.axvline(x=best_epoch, color=highlight_color, linestyle='--', linewidth=2.5, alpha=0.9, label=f'Best Model (Epoch {best_epoch})')
ax11.scatter(best_epoch, best_eval_acc, color=best_color, s=200, zorder=10, 
           edgecolors='black', linewidth=2, marker='*', label=f'Best Eval Acc: {best_eval_acc:.2f}%')
ax11.axvspan(best_epoch-0.5, best_epoch+0.5, alpha=0.2, color=highlight_color)
ax11.set_xlabel('Epoch', fontsize=12)
ax11.set_ylabel('Accuracy (%)', fontsize=12)
ax11.set_title('Accuracy with Moving Average', fontsize=14, fontweight='bold')
ax11.grid(True, alpha=0.3)
ax11.legend(loc='lower right', fontsize=9)
ax11.set_xticks(np.arange(1, 41, 4))

# Gaze loss evolution
ax12.plot(epochs, train_loss_gaze, 'o-', color=gaze_color, linewidth=2, markersize=4, label='Train Gaze Loss')
ax12.plot(epochs, eval_loss_gaze, 's-', color=gaze_color, linewidth=1.5, markersize=3, label='Eval Gaze Loss', alpha=0.6)
ax12.axhline(y=np.mean(train_loss_gaze), color=gaze_color, linestyle='--', alpha=0.5, label=f'Train Mean: {np.mean(train_loss_gaze):.4f}')
ax12.axhline(y=np.mean(eval_loss_gaze), color=gaze_color, linestyle=':', alpha=0.5, label=f'Eval Mean: {np.mean(eval_loss_gaze):.4f}')
ax12.set_xlabel('Epoch', fontsize=12)
ax12.set_ylabel('Gaze Loss', fontsize=12)
ax12.set_title('Gaze Loss Evolution (Auxiliary Task)', fontsize=14, fontweight='bold')
ax12.grid(True, alpha=0.3)
ax12.legend(loc='upper right', fontsize=9)
ax12.set_xticks(np.arange(1, 41, 4))

# Performance plateau analysis
ax13.plot(epochs, eval_acc, 's-', color=eval_color, linewidth=2, markersize=5, label='Evaluation Accuracy')
# Highlight plateau regions
plateau_start = 20
plateau_end = 40
plateau_acc = eval_acc[plateau_start-1:plateau_end]
plateau_mean = np.mean(plateau_acc)
plateau_std = np.std(plateau_acc)
ax13.axhspan(plateau_mean - plateau_std, plateau_mean + plateau_std, alpha=0.2, color=eval_color, label=f'Plateau: {plateau_mean:.1f}% ± {plateau_std:.1f}%')
ax13.axhline(y=plateau_mean, color=eval_color, linestyle='--', alpha=0.5)
ax13.axvspan(plateau_start-0.5, plateau_end+0.5, alpha=0.1, color='gray', label='Convergence Phase')
ax13.scatter(best_epoch, best_eval_acc, color=best_color, s=150, zorder=10, 
           edgecolors='black', linewidth=2, marker='*')
ax13.set_xlabel('Epoch', fontsize=12)
ax13.set_ylabel('Evaluation Accuracy (%)', fontsize=12)
ax13.set_title(f'Convergence Analysis (Plateau: Epochs {plateau_start}-{plateau_end})', fontsize=14, fontweight='bold')
ax13.grid(True, alpha=0.3)
ax13.legend(loc='lower right', fontsize=9)
ax13.set_xticks(np.arange(1, 41, 4))
ax13.set_ylim([70, 80])

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.show()

# Print detailed analysis
print("="*90)
print("TRAINING ANALYSIS REPORT")
print("="*90)
print("\n1. BEST MODEL PERFORMANCE:")
print(f"   • Best epoch: {best_epoch}")
print(f"   • Evaluation accuracy: {best_eval_acc:.2f}%")
print(f"   • Evaluation loss: {best_eval_loss:.4f}")
print(f"   • Improvement from start: {best_eval_acc - eval_acc[0]:.2f}%")
print(f"   • Macro F1-score: 0.7409 (from classification report)")

print("\n2. TRAINING DYNAMICS:")
print(f"   • Training accuracy range: {min(train_acc):.1f}% - {max(train_acc):.1f}%")
print(f"   • Evaluation accuracy range: {min(eval_acc):.1f}% - {max(eval_acc):.1f}%")
print(f"   • Train-eval gap: {np.mean(train_acc) - np.mean(eval_acc):.1f}% (avg)")
print(f"   • Classification loss dominant: {np.mean(train_loss_cls)/np.mean(train_loss_total)*100:.1f}% of total loss")

print("\n3. MULTI-TASK LEARNING ANALYSIS:")
print(f"   • Gaze loss stable: {np.mean(train_loss_gaze):.4f} ± {np.std(train_loss_gaze):.4f}")
print(f"   • CLS/Gaze loss ratio: {np.mean(train_loss_cls)/np.mean(train_loss_gaze):.2f}:1")
print(f"   • Gaze task well-learned early (epoch 3-4)")

print("\n4. CONVERGENCE PATTERN:")
print(f"   • Early training (epochs 1-10): Rapid improvement with high LR")
print(f"   • Middle phase (epochs 11-19): Steady improvement with reduced LR")
print(f"   • Convergence phase (epochs 20-40): Plateau around {plateau_mean:.1f}% accuracy")
print(f"   • Best performance achieved after LR reduction to 3e-6")

print("\n5. LEARNING RATE SCHEDULE EFFECT:")
lr_changes = [(10, "3e-4 → 3e-5"), (19, "3e-5 → 3e-6"), (24, "3e-6 → 3e-7"), 
              (27, "3e-7 → 3e-8"), (33, "3e-8 → 3e-9")]
print("   • Stepwise decay schedule:")
for epoch, change in lr_changes:
    print(f"     Epoch {epoch}: {change}")

print("\n6. RECOMMENDATIONS:")
print("   • Model converged well with current training strategy")
print("   • Best model saved at epoch 20 (77.47% accuracy)")
print("   • Consider early stopping around epoch 25-30 to prevent overfitting")
print("   • Multi-task approach effective - gaze loss stabilized early")
print("   • For future runs: Try different gaze loss weighting or schedule")
print("="*90)