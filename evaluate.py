from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay, DistanceMetric
import matplotlib.pyplot as plt

def evaluate(Y_test, Y_pred_float, flag=0):
    accuracy = accuracy_score(Y_test, Y_pred_float)
    precision = precision_score(Y_test, Y_pred_float, average='weighted')
    recall = recall_score(Y_test, Y_pred_float, average='weighted')
    f1 = f1_score(Y_test, Y_pred_float, average='weighted')
    # Confusion Matrix
    if flag:
        conf_matrix = confusion_matrix(Y_test, Y_pred_float)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
        disp.plot(cmap=plt.cm.Blues)
        plt.show()
        # Accuracy
        print("Accuracy:", accuracy)

        # Precision
        print("Precision:", precision)

        # Recall
        print("Recall:", recall)

        # F1 Score
        print("F1 Score:", f1)