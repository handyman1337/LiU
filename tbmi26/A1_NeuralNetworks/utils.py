from IPython import display
from matplotlib import pyplot as plt
import numpy as np
from scipy import io as sio


def loadDataset(datasetNr):
    """
    Loads specific dataset.

    Samples are in the 1st dimension (rows), and features in the
    2nd dimension. This convention must be consistent throughout the
    assignment; otherwise the plot code will break.

    Args:
        datasetNr (int [1-4]): Dataset to load.

    Returns:
        X (array): Data samples.
        D (array): Neural network target values.
        L (array): Class labels.
    """

    if not (1 <= datasetNr and datasetNr <= 4):
        raise ValueError("Unknown dataset number")

    data = sio.loadmat("Data/lab_data.mat")
    X = data[f"X{datasetNr}"]
    D = data[f"D{datasetNr}"]
    L = data[f"L{datasetNr}"].squeeze()

    return X.astype(float), D.astype(float), L.astype(int)


def plotDatasets():
    """
    Plots the datasets used in the assignment.
    """

    plotStrings = ["r.", "g.", "b."]
    c = "xo+*sd"

    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    subplots = fig.subplots(2, 2)

    for (d, ax) in enumerate(subplots[:, :].flat):

        [X, _, L] = loadDataset(d + 1)

        # Plot first three datasets
        if d + 1 <= 3:
            ax.invert_yaxis()
            ax.set_title(f"Dataset {d+1}")
            for label in range(3):
                ind = (L == label).squeeze()
                ax.plot(X[ind,0], X[ind,1], plotStrings[label])
                # ax.plot(X[ind, 0], X[ind, 1], c[i])
                # ax.plot(X[ind, 0], X[ind, 1], ".")
        # Plot fourth dataset
        else:
            gridspec = ax.get_subplotspec().get_gridspec()
            ax.remove()
            subfig = fig.add_subfigure(gridspec[1, 1])
            subplots2 = subfig.subplots(4, 4)
            for (i, ax2) in enumerate(subplots2[:, :].flat):
                ax2.imshow(X[i].reshape(8, 8), cmap="gray")
                ax2.set_title(f"Class {L[i]}")
                ax2.set_axis_off()
            subfig.suptitle("Dataset 4")
    plt.show()        


def splitData(X, D, L, testFraction, seed=None):
    """
    Splits data into training and test portions.

    Args:
        X (array): Data samples.
        D (array): Neural network target values.
        L (array): Class labels.
        testFraction (float [0-1]): Fraction of data used for testing.
        seed (int): Used to enable reliable tests.

    Returns:
        XTrain (array): Training portion of X.
        DTrain (array): Training portion of D.
        LTrain (array): Training portion of L.
        XTest (array): Test portion of X.
        DTest (array): Test portion of D.
        LTest (array): Test portion of L.
    """

    nSamples = X.shape[0]

    if seed is None:
        perm = np.random.permutation(nSamples)
    else:
        perm = np.random.RandomState(seed=seed).permutation(nSamples)

    iTrain = sorted(perm[int(testFraction * nSamples) :])
    iTest = sorted(perm[: int(testFraction * nSamples)])

    return X[iTrain], D[iTrain], L[iTrain], X[iTest], D[iTest], L[iTest]


def splitDataEqualBins(X, D, L, nBins):
    """
    Splits data into separate equal-sized bins.

    Args:
        X (array): Data samples.
        D (array): Training targets for X.
        L (array): Data lables for X.
        nBins (int): Number of bins to split into.

    Returns:
        XBins (list): Output bins with data from X.
        DBins (list): Output bins with data from D.
        LBins (list): Output bins with data from L.
    """

    labels, counts = np.unique(L, return_counts=True)
    nLabels = labels.shape[0]

    nSamplesPerLabelPerBin = counts.min() // nBins

    # Get class labels
    labelInds = {}
    for label in labels:
        labelInds[label] = np.flatnonzero(L == label)
        np.random.shuffle(labelInds[label])

    XBins, DBins, LBins = [], [], []
    for m in range(nBins):
        sampleInds = np.concatenate(
            [a[m * nSamplesPerLabelPerBin : (m + 1) * nSamplesPerLabelPerBin] for a in labelInds.values()], axis=0,
        )

        XBins.append(X[sampleInds])
        DBins.append(D[sampleInds])
        LBins.append(L[sampleInds])

    return XBins, DBins, LBins


def splitDataBins(X, D, L, nBins):
    """
    Splits data into separate equal-sized bins.

    Args:
        X (array): Data samples.
        D (array): Training targets for X.
        L (array): Data lables for X.
        nBins: Number of bins to split into.

    Returns:
        XBins (list): Output bins with data from X.
        DBins (list): Output bins with data from D.
        LBins (list): Output bins with data from L.
    """

    nSamplesPerBin = X.shape[0] // nBins

    I = np.random.permutation(X.shape[0])

    XBins, DBins, LBins = [], [], []
    for b in range(nBins):
        sampleInds = I[b * nSamplesPerBin : (b + 1) * nSamplesPerBin]

        if X is not None:
            XBins.append(X[sampleInds])
        if D is not None:
            DBins.append(D[sampleInds])
        if L is not None:
            LBins.append(L[sampleInds])
    
    return XBins, DBins, LBins


def getCVSplit(XBins,DBins,LBins,nBins,i):
    """
    Combine data bins into training and validation sets
    for cross validation.

    Args:
        XBins (list of arrays): Binned data samples.
        DBins (list of arrays): Binned training targets for X.
        LBins (list of arrays): Binned lables for X.
        nBins (int): Number of bins in X, D, and L.
        i (int): Current cross-validation iteration.

    Returns:
        XTrain (array): Cross validation training data.
        DTrain (array): Cross validation training targets.
        LTrain (array): Cross validation training labels.
        XVal (array): Cross validation validation data.
        DVal (array): Cross validation validation targets.
        LVal (array): Cross validation validation labels.
    """
    
    if XBins is None:
        XTrain = None
        XVal = None
    else:
        XTrain = np.concatenate([XBins[j] for j in np.arange(nBins) if j != i])
        XVal = XBins[i]
        
    if DBins is None:
        DTrain = None
        DVal = None
    else:
        DTrain = np.concatenate([DBins[j] for j in np.arange(nBins) if j != i])
        DVal = DBins[i]
        
    if LBins is None:
        LTrain = None
        LVal = None
    else:
        LTrain = np.concatenate([LBins[j] for j in np.arange(nBins) if j != i])
        LVal = LBins[i]
    
    return XTrain, DTrain, LTrain, XVal, DVal, LVal


def plotResultsCV(meanAccs, kBest):
    """
    Plot accuracies and optimal k from the cross validation.

    Args:
        meanAccs (array): Array of mean accuracies for different k values.
        kBest (int): The value of k with the best mean accuracy.
    """
    kBestAcc = meanAccs[kBest-1]
    kMax = np.size(meanAccs)
    
    plt.figure()
    plt.plot(np.arange(1, kMax+1), meanAccs, "k.-", label="Avg. val. accuracy")
    plt.plot(kBest, kBestAcc, 'bo', label=f"Max avg. val. accuracy, k={kBest}")
    plt.grid()
    plt.legend()
    plt.title(f'Maximum average cross-validation accuracy: {kBestAcc:.4f} for k = {kBest}')
    plt.ylabel("Accuracy")
    plt.xlabel("k")
    plt.show()

    
def _plotData(X, L, LPred):
    """
    Plot dataset 1, 2, or 3. Indicates correct and incorrect label predictions
    as green and red respectively.

    Args:
        X (array): Data samples.
        L (array): True labels for X.
        LPred (array): Predicted labels for X.
    """

    c = "xo+*sd"

    for label in range(3):
        correctInd = (L == label) & (L == LPred)
        errorInd = (L == label) & (L != LPred)
        plt.plot(X[correctInd, 0], X[correctInd, 1], "g" + c[label])
        plt.plot(X[errorInd, 0], X[errorInd, 1], "r" + c[label])


def plotResultsDots(XTrain, LTrain, LPredTrain, XTest, LTest, LPredTest, classifierFunction):
    """
    Plot training and test prediction for datasets 1, 2, or 3.

    Indicates correct and incorrect label predictions, and plots the
    prediction fields as the background color.

    Args:
        XTrain (array): Training data samples.
        LTrain (array): True labels for XTrain.
        LPredTrain (array): Predicted labels for XTrain.
        XTest (array): Test data samples.
        LTest (array): True labels for XTest.
        LPredTest (array): Predicted labels for XTest.
        classifierFunction (function): Function that takes data samples as input
            and outputs predicted labels. Used to compute the prediction fields.
    """

    # Create background meshgrid for plotting label fields
    # Change nx and ny to set the resolution of the fields
    nx = 150
    ny = 150

    xMin = np.min((XTrain[:, 0].min(), XTest[:, 0].min())) - 1
    xMax = np.max((XTrain[:, 0].max(), XTest[:, 0].max())) + 1
    yMin = np.min((XTrain[:, 1].min(), XTest[:, 1].min())) - 1
    yMax = np.max((XTrain[:, 1].max(), XTest[:, 1].max())) + 1

    xi = np.linspace(xMin, xMax, nx)
    yi = np.linspace(yMin, yMax, ny)

    XI, YI = np.meshgrid(xi, yi)

    # Setup data depending on classifier type
    XGrid = np.column_stack((XI.flatten(), YI.flatten()))
    LGrid = classifierFunction(XGrid).reshape((nx, ny))

    # Plot training data
    plt.figure(figsize=(16,6))
    plt.subplot(1,2,1)
    plt.imshow(
        LGrid, extent=(xMin, xMax, yMin, yMax), cmap="gray", aspect="auto", origin="lower",
    )
    _plotData(XTrain, LTrain, LPredTrain)
    plt.gca().invert_yaxis()
    plt.title(
        f"Training data results (green ok, red error)" + 
        "\n" + 
        f"Error = {100*np.mean(LTrain!=LPredTrain):.2f}% ({np.sum(LTrain!=LPredTrain)} of {LTrain.shape[0]})"
    )

    # Plot test data
    plt.subplot(1,2,2)
    plt.imshow(
        LGrid, extent=(xMin, xMax, yMin, yMax), cmap="gray", aspect="auto", origin="lower",
    )
    _plotData(XTest, LTest, LPredTest)
    plt.gca().invert_yaxis()
    plt.title(
        f"Test data results (green ok, red error)" + 
        "\n" + 
        f"Error = {100*np.mean(LTest!=LPredTest):.2f}% ({np.sum(LTest!=LPredTest)} of {LTest.shape[0]})"
    )

    # Plot
    plt.show()
    
def plotResultsDotsGradient(XTrain, LTrain, LPredTrain, XTest, LTest, LPredTest, classifierFunction):
    """
    Plot training and test prediction for datasets 1, 2, or 3 as a gradient field.

    Indicates correct and incorrect label predictions, and plots the
    prediction fields as the background color.

    Args:
        XTrain (array): Training data samples.
        LTrain (array): True labels for XTrain.
        LPredTrain (array): Predicted labels for XTrain.
        XTest (array): Test data samples.
        LTest (array): True labels for XTest.
        LPredTest (array): Predicted labels for XTest.
        classifierFunction (function): Function that takes data samples as input
            and outputs predicted labels. Used to compute the prediction fields.
    """

    # Create background meshgrid for plotting label fields
    # Change nx and ny to set the resolution of the fields
    nx = 150
    ny = 150

    xMin = np.min((XTrain[:, 0].min(), XTest[:, 0].min())) - 1
    xMax = np.max((XTrain[:, 0].max(), XTest[:, 0].max())) + 1
    yMin = np.min((XTrain[:, 1].min(), XTest[:, 1].min())) - 1
    yMax = np.max((XTrain[:, 1].max(), XTest[:, 1].max())) + 1

    xi = np.linspace(xMin, xMax, nx)
    yi = np.linspace(yMin, yMax, ny)

    XI, YI = np.meshgrid(xi, yi)

    # Setup data depending on classifier type
    XGrid = np.column_stack((XI.flatten(), YI.flatten()))
    YGrid = classifierFunction(XGrid)
    PGrid = np.exp(YGrid) / np.sum(np.exp(YGrid), axis=1, keepdims=True)
    PGrid = np.clip((PGrid) * 1.6 - 0.5, 0, 1) # For color adjustment
    
    # Plot training data
    plt.figure(figsize=(16,6))
    plt.subplot(1,2,1)
    for L in np.unique(LTrain):
        plt.imshow(
            np.ones((nx,ny)), extent=(xMin, xMax, yMin, yMax), cmap=["Reds", "Greens", "Blues"][L], vmin=0, vmax=1,
            aspect="auto", origin="lower", alpha=PGrid[:,L].reshape((nx,ny))
        )
    _plotData(XTrain, LTrain, LPredTrain)
    plt.gca().invert_yaxis()
    plt.title(
        f"Training data results (green ok, red error)" + 
        "\n" + 
        f"Error = {100*np.mean(LTrain!=LPredTrain):.2f}% ({np.sum(LTrain!=LPredTrain)} of {LTrain.shape[0]})"
    )

    # Plot test data
    plt.subplot(1,2,2)
    for L in np.unique(LTest):
        plt.imshow(
            np.ones((nx,ny)), extent=(xMin, xMax, yMin, yMax), cmap=["Reds", "Greens", "Blues"][L], vmin=0, vmax=1,
            aspect="auto", origin="lower", alpha=PGrid[:,L].reshape((nx,ny))
        )
    _plotData(XTest, LTest, LPredTest)
    plt.gca().invert_yaxis()
    plt.title(
        f"Test data results (green ok, red error)" + 
        "\n" + 
        f"Error = {100*np.mean(LTest!=LPredTest):.2f}% ({np.sum(LTest!=LPredTest)} of {LTest.shape[0]})"
    )

    # Plot
    plt.show()


def _plotCase(X, L):
    """
    Simple plot of data. Can only be used with dataset 1, 2, and 3.

    Args:
        X (array): Data samples.
        L (array): True labels for X.
    """

    plotStrings = ["r.", "g.", "b."]

    for label in range(3):
        ind = (L == label).squeeze()
        plt.plot(X[ind, 0], X[ind, 1], plotStrings[label])
    plt.gca().invert_yaxis()


def plotIsolines(X, L, classifierFunction):
    """
    Plot isolevels of neural network output for datasets 1-3.

    Args:
        X (array): Data samples.
        L (array): True labels for X.
        classifierFunction (function): Function that takes data samples as input
            and outputs predicted labels. Used to compute the prediction fields.
    """

    cmaps = ["Reds", "Greens", "Blues"]

    # Create background meshgrid for plotting label fields
    # Change nx and ny to set the resolution of the fields
    nx = 150
    ny = 150

    xMin, yMin = X.min(axis=0) - 1
    xMax, yMax = X.max(axis=0) + 1

    xi = np.linspace(xMin, xMax, nx)
    yi = np.linspace(yMin, yMax, ny)

    XI, YI = np.meshgrid(xi, yi)

    # Setup data depending on classifier type
    XGrid = np.column_stack((XI.flatten(), YI.flatten()))
    YGrid = classifierFunction(XGrid)

    # Plot training data
    plt.figure()
    _plotCase(X, L)
    for i in range(YGrid.shape[1]):
        a = YGrid[:, i].reshape((nx, ny))
        plt.contour(XI, YI, a, np.linspace(0, 2, 6), cmap=cmaps[i])
        plt.contour(XI, YI, a, [1], colors="black")
    # plt.gca().invert_yaxis()

    # Plot
    plt.show()


def plotResultsOCR(X, L, LPred):
    """
    Plots the results using the 4th dataset (OCR). Selects a
    random set of 16 samples each time.

    Args:
        X (array): Data samples.
        L (array): True labels for X.
        LPred (array): Predicted labels for X.
    """

    L = L.astype(int)
    LPred = LPred.astype(int)

    # Create random sort vector
    ord = np.random.permutation(X.shape[0])

    plt.figure(figsize=(6, 6), tight_layout=True)

    # Plot 16 samples
    for n in range(16):
        idx = ord[n]
        plt.subplot(4, 4, n + 1)
        plt.imshow(X[idx].reshape((8, 8)), cmap="gray")
        plt.title("$L_{true}=$" + f"{L[idx]}" + "\n $L_{pred}=$" + f"{LPred[idx]}")
        plt.axis("off")
    plt.suptitle("Random selection of samples")
    plt.show()


def plotConfusionMatrixOCR(X, L, LPred):
    """
    Plots a 10x10 matrix of MNIST digit examples, where the rows
    correspond to predicted labels and the columns to actual labels.

    Args:
        X (array): Data samples.
        L (array): True labels for X.
        LPred (array): Predicted labels for X.
    """
    canvas = np.zeros((107, 107))
    for i in range(10):
        for j in range(10):

            I = np.flatnonzero((LPred == i) & (L == j))

            if I.size != 0:
                canvas[i * 11 : i * 11 + 8, j * 11 : j * 11 + 8] = X[np.random.choice(I)].reshape((8, 8))

    plt.figure(figsize=(6,6))
    plt.imshow(canvas, cmap="gray")
    plt.xticks(ticks=np.arange(3, 107, 11), labels=np.arange(10))
    plt.yticks(ticks=np.arange(3, 107, 11), labels=np.arange(10))
    plt.tick_params(bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.xlabel("Actual class")
    plt.ylabel("Predicted class")
    plt.gca().xaxis.set_label_position("top")
    plt.title("Examples cases from the confusion matrix")
    plt.show()


def plotProgressGraphs(axU, axL, metrics, n=None):
    """
    Plots the training and test losses and accuracies.
    Args:
        axU (matplotlib.axes.Axes): The axes to plot the losses on.
        axL (matplotlib.axes.Axes): The axes to plot the accuracies on.
        metrics (dict): Dictionary containing the following keys:
            - lossTrain: Training loss values.
            - lossTest: Test loss values.
            - accTrain: Training accuracy values.
            - accTest: Test accuracy values.
        n (int, optional): Number of iterations to plot. If None, plots all iterations.
    """

    numIterations = metrics["lossTrain"].shape[0]
    if n is None:
        n = numIterations

    minErrTest = np.nanmin(metrics["lossTest"][:n])
    minErrTestInd = np.nanargmin(metrics["lossTest"][:n])
    maxAccTest = np.nanmax(metrics["accTest"][:n])
    maxAccTestInd = np.nanargmax(metrics["accTest"][:n])
    
    axU.cla()
    axU.semilogy(metrics["lossTrain"][:n], "k", linewidth=1.5, label="Training Loss")
    axU.semilogy(metrics["lossTest"][:n], "r", linewidth=1.5, label="Test Loss")
    axU.semilogy(minErrTestInd, minErrTest, "bo", linewidth=1.5, label="Min Test Loss")
    axU.set_xlim([0, numIterations])
    axU.grid("on")
    axU.legend()
    axU.set_xlabel("Epochs")
    axU.set_ylabel("Loss (mean-squared error)")

    axL.cla()
    axL.plot(metrics["accTrain"][:n], "k", linewidth=1.5, label="Training Accuracy")
    axL.plot(metrics["accTest"][:n], "r", linewidth=1.5, label="Test Accuracy")
    axL.plot(maxAccTestInd, maxAccTest, "bo", linewidth=1.5, label="Max Test Accuracy")
    axL.set_xlim([0, numIterations])
    axL.grid("on")
    axL.legend()
    axL.set_xlabel("Epochs")
    axL.set_ylabel("Accuracy")

def plotNetwork(ax, W, B, cmap="coolwarm"):
    """
    Plots the progress of a multi-layer neural network training and the network weights.

    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        W (list of arrays): List of weight matrices for each layer.
        B (list of arrays): List of bias vectors for each layer.
        metrics (dict): Dictionary containing training and test metrics.
        cmap (str): Colormap to use for visualizing weights.
    """

    ax.axis("off")

    # Colormap
    cm = plt.cm.get_cmap(cmap)

    # Calculate the maximum absolute value of weights and biases for color normalization, then perform normalization
    vmap = np.max([np.abs(w).max() for w in W + B])
    W = [w / (2 * vmap) + 0.5 for w in W]
    B = [b / (2 * vmap) + 0.5 for b in B]

    # Get maximum layer width
    nMax = max([w.shape[0] + 1 for w in W] + [W[-1].shape[1]])
    p = 2  # Padding for nice plots

    # Loop over each layer to plot weights
    for i in range(len(W)):
        # Get weights and biases for the current, and concatenate them
        Wi = np.concatenate((W[i], B[i]), axis=0)
        
        # Calculate y-positions of nodes for current and next layer, accounting for bias nodes.
        # The y-positions are evenly spaced and centered vertically.
        nIn, nOut = Wi.shape
        if i < len(W) - 1:
            nOut = nOut + 1  # Account for bias node in next layer (except for last layer)
        # Linearly spaced y-positions with some padding, for nice plots
        yIn = np.linspace(-(nMax - 1) / 2, (nMax - 1) / 2, nIn+2*p)[p:-p]
        yOut = np.linspace(-(nMax - 1) / 2, (nMax - 1) / 2, nOut+2*p)[p:-p]

        # Calculate x-positions of current and next layer
        xIn = i / len(W)
        xOut = (i + 1) / len(W)

        # Plot neural network weights
        for j in range(nIn):
            for k in range(nOut):
                # Skip bias node in next layer
                if (i < len(W) - 1) and (k == nOut - 1):
                    continue
                ax.plot(
                    [xIn, xOut],
                    [yIn[j], yOut[k]],
                    color=cm(Wi[j, k]),
                    lw=5,
                    marker="o",
                    markersize=20,
                    markerfacecolor="w",
                    markeredgecolor="k",
                )

        # Plot labels for the current layer, if not the first layer
        if (i == 0):
            continue
        for j in range(nIn - 1):
            if len(W) > 2:
                ax.text(xIn - 0.1, yIn[j], f"$U_{{{j}}}^{{{i}}}$", fontsize=16, verticalalignment="center")
            else:
                ax.text(xIn - 0.1, yIn[j], f"$U_{{{j}}}$", fontsize=16, verticalalignment="center")   
        ax.text(xIn - 0.1, yIn[-1], "1", fontsize=16, verticalalignment="center")

    # Plot inputs labels for the first layer
    nIn = W[0].shape[0] + 1 # Account for bias node
    yIn = np.linspace(-(nMax - 1) / 2, (nMax - 1) / 2, nIn+2*p)[p:-p]
    xIn = 0
    for j in range(nIn - 1):
        ax.text(xIn - 0.1, yIn[j], f"$X_{{{j}}}$", fontsize=16, verticalalignment="center")
    ax.text(xIn - 0.1, yIn[-1], "1", fontsize=16, verticalalignment="center")

    # Plot output labels for the last layer
    nOut = W[-1].shape[1]
    yOut = np.linspace(-(nMax - 1) / 2, (nMax - 1) / 2, nOut+2*p)[p:-p]
    xOut = 1
    for j in range(nOut):
        ax.text(xOut + 0.05, yOut[j], f"$Y_{{{j}}}$", fontsize=16, verticalalignment="center")
    
    plt.title("Network weights")

    # Invert y axis
    ax.invert_yaxis()

    # Colorbar
    norm = plt.cm.ScalarMappable(norm=None, cmap=cm)
    norm.set_clim(-vmap, vmap)
    ax.figure.colorbar(norm, ax=ax, location="right")


def plotOutputWeightsOCR(W, cmap="coolwarm"):
    """
    Plots the output weights for OCR tasks.
    Note that this only works for single layer networks since it
    assumes a direct connection from input pixels to output digits.

    Args:
        W (array): Weight matrix of shape (64, 10).
        cmap (str): Colormap to use for visualizing weights.
    """

    # Calculate maximum absolute value of weights for color normalization
    vmax = np.abs(W).max()

    for i in range(10):
        w = W[:, i].reshape(8, 8)
        
        plt.subplot(2, 10, i + 1 + 5 * (i // 5))
        plt.cla()
        plt.axis("off")
        plt.imshow(w, vmin=-vmax * 0.8, vmax=vmax * 0.8, cmap=cmap)
        plt.title(i)

    plt.subplot(2, 10, 3)
    plt.title("Network weights for each digit (blue: positive, red: negative) \n\n 2")


def plotTrainingProgress(fig, W, B, metrics, cmap="coolwarm", n=None, mode="network"):
    """
    Plots the progress of a multi-layer neural network training and the network weights.
    Args:
        fig (matplotlib.figure.Figure): The figure to plot on.
        W (list of arrays): List of weight matrices for each layer.
        B (list of arrays): List of bias vectors for each layer.
        metrics (dict): Dictionary containing training and test metrics.
        cmap (str): Colormap to use for visualizing weights.
        n (int, optional): Current epoch number.
        mode (str): Plotting mode
            - "network": Plots the network weights and training progress graphs.
            - "ocr": Plots the output weights for OCR tasks.
            - "graphs": Plots only the training progress graphs.
    """

    # Check if W and B are lists of weights and biases, else convert to lists
    if not isinstance(W, list):
        W = [W]
    if not isinstance(B, list):
        B = [B]

    # Check that W and B have the same length
    if len(W) != len(B):
        raise ValueError("The number of weight matrices must match the number of bias vectors.")

    # Clear the figure
    plt.clf()

    # Plot based on mode
    if mode == "network":
        # Determine subplot layout based on number of layers
        if len(W) < 4:
            axN = plt.subplot(2, 3, (1, 4))
            axU = plt.subplot(2, 3, (2, 3))
            axL = plt.subplot(2, 3, (5, 6))
        elif len(W) < 7:
            axN = plt.subplot(2, 4, (1, 6))
            axU = plt.subplot(2, 4, (3, 4))
            axL = plt.subplot(2, 4, (7, 8))
        else:
            axN = None
            axU = plt.subplot(2, 1, 1)
            axL = plt.subplot(2, 1, 2)
        
        # Plot network weights and training progress graphs
        if axN is not None:
            plotNetwork(axN, W, B, cmap=cmap)
        plotProgressGraphs(axU, axL, metrics, n=n)

    elif mode == "ocr":
        if (len(W) == 1):
            plotOutputWeightsOCR(W[-1], cmap=cmap)
            axU = plt.subplot(2, 10, (6, 10))
            axL = plt.subplot(2, 10, (16, 20))
        else:
            axU = plt.subplot(2, 1, 1)
            axL = plt.subplot(2, 1, 2)
        plotProgressGraphs(axU, axL, metrics, n=n)
    else:
        raise ValueError("Unknown plotting mode: " + mode)

    # Set super title with current epoch
    if n is not None:
        plt.suptitle(f"Training progress at epoch {n}")
    
    # Render the updated figure
    display.display(fig)
    display.clear_output(wait=True)


def checkExpectedResults(Y, L, dW, dB, W_new, B_new, Yexp, Lexp, dWexp, dBexp, W_new_exp, B_new_exp, decimals=2):
    """
    Compares computed results to expected results and prints whether they are correct.
    A special warning is printed if the gradients are correct up to a scale factor
    equal to the number of samples, or 2, to check for common mistakes in the gradient
    calculations.

    Args:
        Y (array): Computed forward pass output.
        L (array): Computed predicted labels.
        dW (array): Computed weight gradients.
        dB (array): Computed bias gradients.
        W_new (array): Computed updated weights.
        B_new (array): Computed updated biases.
        Yexp (array): Expected forward pass output.
        Lexp (array): Expected predicted labels.
        dWexp (array): Expected weight gradients.
        dBexp (array): Expected bias gradients.
        Wnewexp (array): Expected updated weights.
        Bnewexp (array): Expected updated biases.
        decimals (int): Number of decimal places for comparison.
    """

    # Setup list of allowed scale factors for gradient checks
    allowedScaleFactors = {
        "number of samples": 1 / Y.shape[0],
        "2": 2,
        "number of samples and 2": 2 / Y.shape[0],
    }
    
    def printHelper(msg, got, expected, allowScale=False):
        # Check for exact match
        if np.allclose(got, expected, atol=10**(-decimals)):
            print(msg + f" are \033[32mcorrect\033[0m.")
            return
        
        # Check for scaled match
        if allowScale:
            scaledGots = [got * scale for scale in allowedScaleFactors.values()]
            for i, scaledGot in enumerate(scaledGots):
                if np.allclose(scaledGot, expected, atol=10**(-decimals)):
                    print(msg + f" are \033[33malmost correct\033[0m (differs by a scale factor of {list(allowedScaleFactors.keys())[i]}).")
                    return
        
        # If no match, print incorrectness message
        print(msg + f" are \033[31mincorrect\033[0m, got {got.round(decimals).tolist()}, expected {expected.round(decimals).tolist()}.")

    # Check network outputs        
    printHelper("Forward pass outputs Y", Y, Yexp)
    printHelper("Predicted labels L", L, Lexp)

    # Check gradients
    if isinstance(dW, list):
        for i in range(len(dW)):
            printHelper(f"Gradients dW{i+1}", dW[i], dWexp[i], allowScale=True)
            printHelper(f"Gradients dB{i+1}", dB[i], dBexp[i], allowScale=True)
    else:
        printHelper("Gradients dW", dW, dWexp, allowScale=True)
        printHelper("Gradients dB", dB, dBexp, allowScale=True)

    # Check updated weights and biases
    if isinstance(W_new, list):
        for i in range(len(W_new)):
            printHelper(f"Updated weights W{i+1}_new", W_new[i], W_new_exp[i])
            printHelper(f"Updated biases B{i+1}_new", B_new[i], B_new_exp[i])
    else:
        printHelper("Updated weights W_new", W_new, W_new_exp)
        printHelper("Updated biases B_new", B_new, B_new_exp)
    