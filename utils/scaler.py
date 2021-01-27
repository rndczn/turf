from sklearn import preprocessing
import numpy as np
import pandas as pd

from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax


def scale(df, cols, scaler=preprocessing.MinMaxScaler, scalers={}):
    dfs = []
    for col in cols:
        arr = np.array(df[col].fillna(-999)).reshape(-1,1)
        if col not in scalers:
            scalers[col] = scaler().fit(arr)
        dfs.append(pd.DataFrame(scalers[col].transform(arr),columns=[col]))

    return pd.concat(dfs, axis=1)

def unskew(df, cols, lambdas={}):
    dfs = []
    for col in cols:
        X_num = df[col]
        X_num = X_num.fillna(X_num.mean())
        if abs(X_num.skew()) > 0.8:
            if col not in lambdas:
                 lambdas[col] = boxcox_normmax(X_num+1)
            dfs.append(boxcox1p(X_num, lambdas[col]))

    return pd.concat(dfs, axis=1)