from scipy import stats
import numpy as np

def compute(sq, q):
    srocc = stats.spearmanr(sq, q)[0]
    krocc = stats.stats.kendalltau(sq, q)[0]
    plcc = stats.pearsonr(sq, q)[0]
    rmse = np.sqrt(((sq - q) ** 2).mean())
    mae = np.abs((sq - q)).mean()

    return srocc, krocc, plcc, rmse, mae