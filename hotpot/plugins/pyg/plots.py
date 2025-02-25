import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(
        targets: np.ndarray,
        predictions: np.ndarray,
        class_names=None,
        figsave_path: str = None,
):
    """
    Plots a confusion matrix given one-hot encoded targets and softmax predictions.

    Args:
        targets (np.ndarray): One-hot encoded target vectors (B×N).
        predictions (np.ndarray): Softmax predicted vectors (B×N).
        class_names (list): List of class names (optional).
        figsave_path (str): path to save the figure.
        calc_accuracy (bool): If True, calculate accuracy.
    """
    # Convert one-hot targets to class indices
    target_indices = np.argmax(targets, axis=1)

    # Convert softmax predictions to class indices
    predicted_indices = np.argmax(predictions, axis=1)

    # Compute the confusion matrix
    conf_matrix = np.zeros((119, 119), dtype=np.int64)

    for t, p in zip(target_indices, predicted_indices):
        conf_matrix[t, p] += 1

    if figsave_path:
        fig, ax = plt.subplots()
        ax.imshow(conf_matrix, norm=plt.Normalize(vmin=conf_matrix.min(), vmax=conf_matrix.max()))

        fig.show()

    return conf_matrix
