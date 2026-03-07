import pandas as pd
import numpy as np


def csvConverter(path, filename):
    df = pd.read_csv(path, sep=';')
    df.to_csv(filename, index=False)
    

def standarizationCal(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x, dtype=float)

    if x.ndim == 2:
        return np.array([standarizationCal(x=x[:, i]) for i in range(x.shape[1])])

    if len(x) == 0:
        raise ValueError("Array is empty")
    
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError("Array must contain numeric values only")
    
    xMean = sum(x) / len(x)
    xStd = np.sqrt(sum((x - xMean)**2) / len(x))
    if xStd == 0:
        raise ValueError("Standard deviation is 0, all values are identical, "
        "and cannot calculate standarization")
    
    zScore = (x - xMean) / xStd
    return zScore

def pcaCal(standardizationX, nComponents):
    samples = standardizationX.shape[0]
    resPCA = None
    covMatrix = (1 / (samples - 1)) * np.dot(standardizationX.T, standardizationX)
    eigenVal, eigenVec = np.linalg.eigh(covMatrix)
    indices = np.argsort(eigenVal)[::-1]
    eigenVecNew = eigenVec[:, indices[0:nComponents]]
    resPCA = standardizationX @ eigenVecNew
    
    return resPCA

