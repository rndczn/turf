from sklearn import preprocessing
import numpy as np
import pandas as pd

from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

embedding_threshold = 16

scaler = preprocessing.StandardScaler

def fit_obj_preproc(X):
    X_obj = X.select_dtypes(['object'])
    cols = X_obj.columns
    voc_size = X_obj.nunique()
    X_obj = X_obj.fillna('NA')

    emb_col = []
    dum_col = []
    label_encoders = {}

    for col in cols:
        if voc_size[col] > embedding_threshold:
            emb_col.append(col)
            le = preprocessing.LabelEncoder()
            le.fit(X_obj[col].unique().tolist() + ['NA'])
            label_encoders[col] = le
        else:
            dum_col.append(col)

    return emb_col, dum_col, label_encoders


def emb_transform(X, label_encoders, emb_col):
    X_emb = X[emb_col].fillna('NA')

    for col in set(emb_col) & set(label_encoders.keys()):
            X_emb[col] = label_encoders[col].transform(X_emb[col])
    
    return X_emb


def match_col(X, columns):
    missing = set(columns) - set(X.columns)
    
    for col in missing:
        X[col] = 0
        
    return X[columns]

def dum_transform(X, dum_col):
    return pd.get_dummies(X[dum_col], dummy_na=True)
    

def fit_num_scaler_preproc(X):
    X_num = X.select_dtypes(exclude=['object'])
    num_col = list(X_num.columns)
    X_num = X[num_col]
    X_num = X_num.fillna(X_num.median())
    
    scalers = {}

    for col in num_col:
        s = scaler()
        arr = np.array(X_num[col]).reshape(-1,1)
        try:
            s.fit(arr)
        except ValueError as e:
            print(col)
            print(e)
        scalers[col] = s
        
    return num_col, scalers

def fit_num_unskew_preproc(X):
    X_num = X.select_dtypes(exclude=['object'])
    num_col = list(X_num.columns)
    X_num = X[num_col]
    X_num = X_num.fillna(X_num.median())
    
    lambdas = {}

    for col in num_col:
        if abs(X_num[col].skew()) > 0.8 and False:
            lambdas[col] = boxcox_normmax(X_num[col]-X_num[col].min()+1)
        
    return num_col, lambdas
    
def num_scaler_transform(X, scalers={}, num_col=[]):
    if not num_col:
        num_col = X.select_dtypes(exclude=['object']).columns
        
    X_num = X[num_col]
    X_num = X_num.fillna(X_num.mean())
    
    for col in set(num_col) & set(scalers.keys()):
        arr = np.array(X_num[col]).reshape(-1,1)
        X_num[col] = scalers[col].transform(arr)
        
    return X_num
    
def num_unskew_transform(X, lambdas={}, num_col=[]):
    if not num_col:
        num_col = X.select_dtypes(exclude=['object']).columns
        
    X_num = X[num_col]
    X_num = X_num.fillna(X_num.mean())
    
    for col in set(num_col) & set(lambdas.keys()):
        X_num[col] =  boxcox1p(X_num[col], lambdas[col])
        
    return X_num   
    
def fit_preproc(X):
    
    emb_col, dum_col, label_encoders = fit_obj_preproc(X)
    num_col, lambdas = fit_num_unskew_preproc(X)
    num_col, scalers = fit_num_scaler_preproc(X)

    return {
        'emb_col': emb_col,
        'dum_col': dum_col,
        'num_col': num_col,
        'label_encoders': label_encoders,
        'scalers': scalers,
        'lambdas': lambdas
    }


def unskew(X, threshold=0.8):
    s = X.skew()
    X[s[np.abs(s)>threshold].index] = X[s[np.abs(s)>threshold].index].apply(lambda x: boxcox1p(x, boxcox_normmax(x+1)))

    return X

class Preprocessor():
    
    def __init__(self):
        self.emb_col = []
        self.dum_col = []
        self.num_col = []
        self.dum_col_ref = []
        
        self.label_encoders = {}
        self.scalers = {}
        self.lambdas = {}

        
    def fit(self, X):
        for k, v in fit_preproc(X).items():
            setattr(self, k, v)
        self.dum_col_ref = list(dum_transform(X, self.dum_col).columns)

    def transform_split(self, X):
        X_num = num_unskew_transform(X, self.lambdas, self.num_col)
        X_num = num_scaler_transform(X_num, self.scalers, self.num_col)
        X_emb = emb_transform(X, self.label_encoders, self.emb_col)
        X_dum = match_col(dum_transform(X, self.dum_col), self.dum_col_ref)
        
        return X_num, X_emb, X_dum