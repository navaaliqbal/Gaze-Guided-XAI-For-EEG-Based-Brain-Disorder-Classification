import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Create figure
fig = plt.figure(figsize=(20, 16))

# 1. Dataset Distribution
ax1 = plt.subplot(4, 4, 1)
splits = ['Train', 'Validation', 'Test']
class_0 = [749, 80, 209]
class_1 = [248, 31, 115]

x = np.arange(len(splits))
width = 0.35

bars1 = ax1.bar(x - width/2, class_0, width, label='Class 0', color='skyblue')
bars2 = ax1.bar(x + width/2, class_1, width, label='Class 1', color='lightcoral')

ax1.set_xlabel('Dataset Split')
ax1.set_ylabel('Number of Samples')
ax1.set_title('Class Distribution Across Splits')
ax1.set_xticks(x)
ax1.set_xticklabels(splits)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add value labels
for bar in bars1 + bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
            f'{int(height)}', ha='center', va='bottom', fontsize=9)

# Calculate percentages
train_total = class_0[0] + class_1[0]
val_total = class_0[1] + class_1[1]
test_total = class_0[2] + class_1[2]

train_pct_0 = (class_0[0]/train_total)*100
train_pct_1 = (class_1[0]/train_total)*100
val_pct_0 = (class_0[1]/val_total)*100
val_pct_1 = (class_1[1]/val_total)*100
test_pct_0 = (class_0[2]/test_total)*100
test_pct_1 = (class_1[2]/test_total)*100

# 2. Class Balance Pie Charts
ax2 = plt.subplot(4, 4, 2)
train_sizes = [class_0[0], class_1[0]]
colors = ['skyblue', 'lightcoral']
wedges, texts, autotexts = ax2.pie(train_sizes, labels=['Class 0', 'Class 1'], 
                                   autopct='%1.1f%%', colors=colors,
                                   startangle=90)
ax2.set_title(f'Train Set: {train_total} samples')
ax2.axis('equal')

ax3 = plt.subplot(4, 4, 3)
val_sizes = [class_0[1], class_1[1]]
wedges, texts, autotexts = ax3.pie(val_sizes, labels=['Class 0', 'Class 1'], 
                                   autopct='%1.1f%%', colors=colors,
                                   startangle=90)
ax3.set_title(f'Validation Set: {val_total} samples')
ax3.axis('equal')

ax4 = plt.subplot(4, 4, 4)
test_sizes = [class_0[2], class_1[2]]
wedges, texts, autotexts = ax4.pie(test_sizes, labels=['Class 0', 'Class 1'], 
                                   autopct='%1.1f%%', colors=colors,
                                   startangle=90)
ax4.set_title(f'Test Set: {test_total} samples')
ax4.axis('equal')

# Training statistics from logs
epochs = list(range(1, 31))
train_loss = [0.8242, 0.7710, 0.7315, 0.7309, 0.7215, 0.7046, 0.6744, 0.6852, 0.6834, 
              0.6821, 0.6677, 0.6803, 0.6757, 0.6859, 0.6549, 0.6755, 0.6625, 0.6727,
              0.6772, 0.6626, 0.6838, 0.6749, 0.6709, 0.6921, 0.6863, 0.6624, 0.6821,
              0.6712, 0.6583, 0.6729]

val_loss = [0.8254, 0.7276, 0.7355, 0.7791, 0.7339, 0.7498, 0.7693, 0.7623, 0.7416,
            0.7463, 0.7574, 0.7838, 0.7478, 0.7643, 0.7400, 0.7315, 0.7606, 0.7550,
            0.7710, 0.7587, 0.7556, 0.7594, 0.7659, 0.7573, 0.7465, 0.7489, 0.7583,
            0.7658, 0.7409, 0.7520]

train_acc = [55.77, 53.46, 67.90, 65.10, 66.30, 65.80, 69.41, 67.80, 68.30, 68.71,
             71.51, 70.21, 69.71, 69.21, 71.31, 69.61, 71.21, 69.71, 70.61, 71.92,
             68.00, 69.21, 70.51, 67.40, 68.10, 71.41, 68.10, 70.71, 71.41, 69.91]

val_acc = [45.05, 60.36, 57.66, 48.65, 63.96, 60.36, 59.46, 62.16, 61.26, 61.26,
           62.16, 57.66, 61.26, 62.16, 60.36, 62.16, 62.16, 62.16, 58.56, 62.16,
           62.16, 62.16, 62.16, 62.16, 61.26, 61.26, 62.16, 62.16, 61.26, 61.26]

# 3. Loss Curves
ax5 = plt.subplot(4, 2, 3)
ax5.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)
ax5.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=4)
ax5.set_xlabel('Epoch')
ax5.set_ylabel('Loss')
ax5.set_title('Training and Validation Loss')
ax5.grid(True, alpha=0.3)
ax5.legend()
ax5.set_xticks(range(1, 31, 2))
ax5.set_ylim(0.6, 0.9)

# Mark best validation loss
best_val_loss_epoch = epochs[val_loss.index(min(val_loss))]
best_val_loss = min(val_loss)
ax5.plot(best_val_loss_epoch, best_val_loss, 'g*', markersize=15, 
         label=f'Best Val Loss: {best_val_loss:.4f} (Epoch {best_val_loss_epoch})')
ax5.legend()

# 4. Accuracy Curves
ax6 = plt.subplot(4, 2, 4)
ax6.plot(epochs, train_acc, 'b-', linewidth=2, label='Training Accuracy', marker='o', markersize=4)
ax6.plot(epochs, val_acc, 'r-', linewidth=2, label='Validation Accuracy', marker='s', markersize=4)
ax6.set_xlabel('Epoch')
ax6.set_ylabel('Accuracy (%)')
ax6.set_title('Training and Validation Accuracy')
ax6.grid(True, alpha=0.3)
ax6.legend()
ax6.set_xticks(range(1, 31, 2))
ax6.set_ylim(40, 75)

# Mark best validation accuracy
best_val_acc = max(val_acc)
best_val_acc_epoch = epochs[val_acc.index(best_val_acc)]
ax6.plot(best_val_acc_epoch, best_val_acc, 'g*', markersize=15,
        label=f'Best Val Acc: {best_val_acc:.2f}% (Epoch {best_val_acc_epoch})')
ax6.legend()

# 5. Learning Rate Schedule
learning_rates = [0.0005] * 5 + [5e-05] * 4 + [5e-06] * 3 + [5e-07] * 3 + [5e-08] * 3 + [5e-09] * 12
ax7 = plt.subplot(4, 2, 5)
ax7.semilogy(epochs, learning_rates, 'g-', linewidth=2.5, marker='D', markersize=6)
ax7.set_xlabel('Epoch')
ax7.set_ylabel('Learning Rate (log scale)')
ax7.set_title('Learning Rate Schedule During Training')
ax7.grid(True, alpha=0.3, which='both')
ax7.set_xticks(range(1, 31, 2))

# 6. Loss Components
epochs_short = epochs[:15]  # First 15 epochs for clarity
train_cls_loss = [0.6952, 0.6901, 0.6576, 0.6555, 0.6478, 0.6309, 0.5992, 0.6080, 
                  0.6063, 0.6048, 0.5904, 0.6029, 0.5981, 0.6083, 0.5774]
train_gaze_loss = [0.5158, 0.3233, 0.2957, 0.3015, 0.2947, 0.2949, 0.3008, 0.3089,
                   0.3085, 0.3090, 0.3093, 0.3095, 0.3104, 0.3104, 0.3100]

ax8 = plt.subplot(4, 2, 6)
x_pos = np.arange(len(epochs_short))
width = 0.35
bars_cls = ax8.bar(x_pos - width/2, train_cls_loss, width, label='Classification Loss', 
                   color='royalblue', alpha=0.7)
bars_gaze = ax8.bar(x_pos + width/2, train_gaze_loss, width, label='Gaze Loss', 
                    color='orange', alpha=0.7)
ax8.set_xlabel('Epoch')
ax8.set_ylabel('Loss Value')
ax8.set_title('Training Loss Components (First 15 Epochs)')
ax8.set_xticks(x_pos)
ax8.set_xticklabels(epochs_short)
ax8.legend()
ax8.grid(True, alpha=0.3)

# 7. Final Performance Comparison
ax9 = plt.subplot(4, 2, 7)
metrics = ['Accuracy', 'Balanced Acc', 'Macro F1', 'Weighted F1']
test_scores = [70.06, 66.82, 66.98, 69.91]  # From final test results
val_scores_best = [63.96, None, None, None]  # Best validation accuracy

x_pos = np.arange(len(metrics))
width = 0.35
bars_test = ax9.bar(x_pos - width/2, test_scores, width, label='Test Set', 
                    color='green', alpha=0.7)
bars_val = ax9.bar(x_pos + width/2, [val_scores_best[0], 0, 0, 0], width, 
                   label='Best Validation', color='red', alpha=0.7)
ax9.set_xlabel('Metric')
ax9.set_ylabel('Score (%)')
ax9.set_title('Final Performance Comparison')
ax9.set_xticks(x_pos)
ax9.set_xticklabels(metrics)
ax9.legend()
ax9.grid(True, alpha=0.3)

# Add value labels
for bar in bars_test:
    height = bar.get_height()
    ax9.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# 8. Confusion Matrix Visualization (simplified)
ax10 = plt.subplot(4, 2, 8)
# From final test classification report
cm_data = np.array([[163, 46],   # TP for class 0, FN for class 0
                    [51, 64]])   # FP for class 1, TP for class 1

im = ax10.imshow(cm_data, cmap='Blues', interpolation='nearest')
ax10.set_title('Test Set Confusion Matrix')
ax10.set_xlabel('Predicted')
ax10.set_ylabel('Actual')
ax10.set_xticks([0, 1])
ax10.set_yticks([0, 1])
ax10.set_xticklabels(['Class 0', 'Class 1'])
ax10.set_yticklabels(['Class 0', 'Class 1'])

# Add text annotations
thresh = cm_data.max() / 2.
for i in range(2):
    for j in range(2):
        ax10.text(j, i, f'{cm_data[i, j]}',
                 ha="center", va="center",
                 color="white" if cm_data[i, j] > thresh else "black",
                 fontweight='bold')

plt.colorbar(im, ax=ax10, fraction=0.046, pad=0.04)

# Add hyperparameter summary
fig.text(0.02, 0.02, 
         f"""HYPERPARAMETER SUMMARY:
         • Learning Rate: 0.0005 (with scheduler)
         • Batch Size: 32
         • Epochs: 30
         • Gaze Weight: 0.25
         • Gaze Loss Type: MSE
         • Gaze Loss Scaling Factor: 9.39
         • Effective Gaze Scale: 2.35
         • Accumulation Steps: 1
         • Validation Split: 10%
         • Data: 22 EEG channels × 15000 timepoints
         • Classes: 0 (Control), 1 (ADHD)
         • Total Samples: Train=997, Val=111, Test=324
         • Class Imbalance: Train (75%/25%), Val (72%/28%), Test (65%/35%)
         • Best Epoch: {best_val_loss_epoch} (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.2f}%)
         • Final Test Acc: 70.06%
         
         Model: Gaze-Guided Attention Training with EEG data
         Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}""",
         fontsize=9, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.7),
         verticalalignment='bottom')

plt.suptitle('Gaze-Guided EEG Attention Training: Comprehensive Analysis', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.05, 1, 0.98])
plt.show()