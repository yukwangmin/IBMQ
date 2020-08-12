import numpy as np

def relativeError(true: float, est: float):
    return (np.abs(true - est) / np.abs(true))


