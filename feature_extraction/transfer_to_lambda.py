import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

def calculate_ppv_npv(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    return ppv, npv

def plot_ppv_npv(hsemotion_smile, final_smile_labels, folder_path):
    thresholds = np.arange(0, 1.05, 0.05)
    ppv_list = []
    npv_list = []

    for threshold in thresholds:
        y_true = []
        y_pred = []
        for timestamp, df in hsemotion_smile.items():
            happiness_scores = df['Happiness'].values
            events_detected = np.convolve(happiness_scores > threshold, np.ones(2, dtype=int), 'valid') == 2
            predicted_label = int(np.any(events_detected))
            ground_truth_label = final_smile_labels.loc[final_smile_labels['Datetime'] == timestamp, 'EventDetected'].values[0]

            y_true.append(ground_truth_label)
            y_pred.append(predicted_label)

        ppv, npv = calculate_ppv_npv(y_true, y_pred)
        ppv_list.append(ppv)
        npv_list.append(npv)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, ppv_list, label='PPV', marker='o')
    plt.plot(thresholds, npv_list, label='NPV', marker='x')
    plt.xlabel('Threshold')
    plt.ylabel('Value')
    plt.title('PPV and NPV vs Threshold')
    plt.legend()
    plt.grid()
    plt.savefig(f"{folder_path}/ppv_npv_plot.png")
    plt.close()

def plot_roc_curve(hsemotion_smile, final_smile_labels, folder_path):
    y_true = []
    y_scores = []

    for timestamp, df in hsemotion_smile.items():
        happiness_scores = df['Happiness'].values
        max_happiness_score = max(happiness_scores)
        ground_truth_label = final_smile_labels.loc[final_smile_labels['Datetime'] == timestamp, 'EventDetected'].values[0]

        y_true.append(ground_truth_label)
        y_scores.append(max_happiness_score)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(f"{folder_path}/roc_curve.png")
    plt.close()

    # Find high-sensitivity threshold above 0.5 with specificity > 30%
    high_sensitivity_threshold = None
    for i in range(len(thresholds)):
        if thresholds[i] > 0.5 and (1 - fpr[i]) > 0.3:
            high_sensitivity_threshold = thresholds[i]
            break

    print(f"High sensitivity threshold above 0.5 with specificity > 30%: {high_sensitivity_threshold}")
