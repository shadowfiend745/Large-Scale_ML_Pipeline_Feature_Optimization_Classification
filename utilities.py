import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def csvConverter(path, filename):
    df = pd.read_csv(path, sep=';')
    df.to_csv(filename, index=False)

def standarizationCal(x, mode, trainMean=None, trainStd=None):
    if not isinstance(x, np.ndarray):
        x = np.array(x, dtype=float)

    if len(x) == 0:
        raise ValueError("Array is empty")
    
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError("Array must contain numeric values only")
    
    if x.ndim == 2:
        if mode == "train":
            trainMean, trainStd, zScore = [], [], []
            for i in range(x.shape[1]):
                z, m, s = standarizationCal(x=x[:, i], mode="train")
                trainMean.append(m)
                trainStd.append(s)
                zScore.append(z)
            
            return np.array(zScore).T, np.array(trainMean), np.array(trainStd)
    
        if mode == "test":
            zScore = []
            for i in range(x.shape[1]):
                zScore.append(standarizationCal(x=x[:, i], mode="test", trainMean=trainMean[i], 
                                                trainStd=trainStd[i]))
                
            return np.array(zScore).T
        
    if mode == "train":
        trainMean = np.mean(x)
        trainStd = np.std(x)
        if trainStd == 0:
            raise ValueError("Standard deviation is 0, all values are identical, "
            "and cannot calculate standarization")
        zScore = (x - trainMean) / trainStd

        return zScore, trainMean, trainStd
    
    if mode == "test":
        if trainMean is None or trainStd is None:
            raise ValueError("Mean or standard deviation from training set are missing")
        zScore = (x - trainMean) / trainStd

        return zScore


def pcaCal(standardizationX, nComponents, mode, eigenVecTop=None):
    samples = standardizationX.shape[0]
    resPCA = None
    if mode == "train":
        covMatrix = (1 / (samples - 1)) * np.dot(standardizationX.T, standardizationX)
        eigenVal, eigenVec = np.linalg.eigh(covMatrix)
        indices = np.argsort(eigenVal)[::-1]
        eigenVecTop = eigenVec[:, indices[0:nComponents]]
        resPCA = standardizationX @ eigenVecTop

        return resPCA, eigenVecTop
    
    if mode == "test":
        if eigenVecTop is None:
            raise ValueError("Need top k eigenvectors to transform test dataset")
        resPCA = standardizationX @ eigenVecTop

        return resPCA

def euclideanDisCal(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knnCal(X_train, y_train, X_test, k):
    predictions = []
    for rowTest in X_test:
        distances = []
        for rowTrain, label in zip(X_train, y_train):
            d = euclideanDisCal(rowTest, rowTrain)
            distances.append((d, label))
        distances.sort(key=lambda x: x[0])
        knnLabels = [label for _, label in distances[:k]]

        prediction = max(set(knnLabels), key=knnLabels.count)
        predictions.append(prediction)

    return predictions

def confusionMatrixGen(y_true, y_prediction):
    classes = np.unique(y_true)
    numClass = len(classes)
    cm = np.zeros((numClass, numClass), dtype=int)

    labelIndMap = {label: i for i, label in enumerate(classes)}
    for actual, predict in zip(y_true, y_prediction):
        row = labelIndMap[actual]
        col = labelIndMap[predict]

        cm[row, col] += 1
    
    return cm

def kSelection(X_train, y_train, X_val, y_val):
    sampleLen = X_train.shape[0]
    sampleLenSqrt = int(np.sqrt(sampleLen))
    kCandidate = np.geomspace(start=1, stop=sampleLenSqrt, num=15).astype(int)
    kCandidate = np.unique([v if v % 2 != 0 else v + 1 for v in kCandidate])

    highestAcc = 0
    bestK = 1
    for k in kCandidate:
        pred = knnCal(X_train=X_train, y_train=y_train, X_test=X_val, k=k)
        accuracy = np.mean(pred == y_val) * 100
        print(f"   k={k}, acc={accuracy:.2f}%")

        if accuracy > highestAcc:
            highestAcc = accuracy
            bestK = k
    
    return bestK

def visualizeCM(confusionMatrix, classes):
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusionMatrix, 
                annot=True,          # show numbers in cells
                fmt='d',             # integer format
                cmap='Blues',        # color scheme
                xticklabels=classes, 
                yticklabels=classes)

    plt.title('KNN Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

