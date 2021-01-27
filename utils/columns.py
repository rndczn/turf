import pandas as pd
import numpy as np
import re

"""
This module aims at separating the different column types
"""

TEXT_THRESHOLD = 10
CATS_THRESHOLD = 10


def types(df, verbose=True):
    text, tags, cats, boolean= [], [], [], []
    cols = df.select_dtypes(exclude=[np.number]).columns.values
    N = max(len(col) for col in cols)
    for col in df.select_dtypes(exclude=[np.number]).columns:
        if df[col].dtype == np.bool:
            l = 1
        else:
            l = df[col].astype(str).fillna('None').str.split().str.len().max()
        n = df[col].nunique()
        if n == 2:
            boolean.append(col)
            if verbose:print(col.ljust(N, ' '), '-> bool', f'(max words: {l}, unique: {n})')
        elif n < CATS_THRESHOLD:
            if verbose:print(col.ljust(N, ' '), '-> cats', f'(max words: {l}, unique: {n})')
            cats.append(col)
        else:
            if l < TEXT_THRESHOLD:
                if verbose:print(col.ljust(N, ' '), '-> tags', f'(max words: {l}, unique: {n})')
                tags.append(col)
            else:
                if verbose:print(col.ljust(N, ' '), '-> text', f'(max words: {l}, unique: {n})')
                text.append(col)

    return {
        'num': list(df.select_dtypes(include=[np.number]).columns.values),
        'text': text,
        'tags': tags,
        'cats': cats,
        'bool': boolean
    }

first_cap_re = re.compile('(.)([A-Z][a-z]+)')
all_cap_re = re.compile('([a-z0-9])([A-Z])')
def to_snake_case(name):
    s1 = first_cap_re.sub(r'\1_\2', name)
    return all_cap_re.sub(r'\1_\2', s1).lower()

def downcast(df):
    d = df.copy()
    for col in df:
        n = d[col].nunique()
        if n < 2:
            d.drop(col,inplace=True,axis=1)
            continue
        if n == 2:
            if df[col].dtype == np.bool:
                continue
            
            a = df[col].iloc[0]
            d[col] = d[col] == a
            continue
        if d[col].dtype == np.int:
            d[col] = pd.to_numeric(d[col],downcast='integer')
            d[col] = pd.to_numeric(d[col],downcast='unsigned')
            continue
        if d[col].dtype == np.float:
            try:
                if (df[col]==df[col].astype(int)).all():
                    d[col] = pd.to_numeric(d[col],downcast='integer')
                    d[col] = pd.to_numeric(d[col],downcast='unsigned')
            except:
                pass
            continue
        if d[col].dtype == 'object':
            if 2 * n <= len(d.index) :
                d[col] = d[col].fillna('MISSING').astype('category')
    return d