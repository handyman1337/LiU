import numpy as np

def calcAccuracy(LPred, LTrue):
    """Calculates prediction accuracy from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Retruns:
        acc (float): Prediction accuracy.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    acc = np.mean(LPred == LTrue) #correct predictions/total
    # ============================================
    return acc


def calcConfusionMatrix(LPred, LTrue):
    """Calculates a confusion matrix from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Returns:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    #Get unique labels and initialize confusion matrix
    labels = np.unique(LTrue)
    cM = np.zeros((len(labels), len(labels)), dtype=int)

    #Dictionary to map label to index in confusion matrix
    label_to_index = {label: i for i, label in enumerate(labels)}

    #Fill confusion matrix (p = predicted, t = true)
    for p, t in zip(LPred, LTrue):
        i = label_to_index[p]
        j = label_to_index[t]
        cM[i, j] += 1
    # ============================================

    return cM


def calcAccuracyCM(cM):
    """Calculates prediction accuracy from a confusion matrix.

    Args:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.

    Returns:
        acc (float): Prediction accuracy.
    """

    # --------------------------------------------
    # === Your code here =========================
    # --------------------------------------------
    #np.trace(cM) = sum of diagonal elements = number of correct predictions
    #np.sum(cM) = total number of predictions
    acc = np.trace(cM) / np.sum(cM)
    # ============================================
    
    return acc
