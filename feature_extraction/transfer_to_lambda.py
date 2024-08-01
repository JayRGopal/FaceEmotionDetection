from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrices(hsemotion_smile, final_smile_labels, folder_path):
    thresholds = [0.8, 0.85, 0.9]
    
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
        
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f'Confusion Matrix at Threshold {threshold}')
        plt.savefig(f"{folder_path}/confusion_matrix_smile_{threshold}.png")
        plt.show()

