import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
  
  fig, (ax1, ax2) = plt.subplots(1, 2)

  ax1.plot(train_losses, label="Training Loss")
  ax1.plot(valid_losses, label="Validation Loss")
  ax1.set_ylabel("Loss")
  ax1.set_xlabel("epoch")
  ax1.set_title("Loss Curve")
  ax1.legend()

  ax2.plot(train_accuracies, label="Training Accuracy")
  ax2.plot(valid_accuracies, label="Validation Accuracy")
  ax2.set_ylabel("Accuracy")
  ax2.set_xlabel("epoch")
  ax2.set_title("Accuracy Curve")
  ax2.legend()

  plt.show()

  pass


def plot_confusion_matrix(results, class_names):
  
  y_true = np.array(list(list(zip(*results))[0])).astype(int)
  y_pred = np.array(list(list(zip(*results))[1])).astype(int)

  d = [0, 1, 2, 3, 4]

  cm = confusion_matrix(y_true, y_pred, labels=d)

  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
  disp.plot()
  plt.show()

  pass
