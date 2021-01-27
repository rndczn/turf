import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from stop_words import get_stop_words

from sklearn import preprocessing

# create french stop-words list
stop_words = list(get_stop_words('fr'))
nltk_words = list(stopwords.words('french'))
stop_words.extend(nltk_words)


# count agregate
def get_count(df, field, by_field):
    tmp = df[[by_field] + [field]].copy()
    tmp[field].fillna('xxx', inplace=True)
    tmp = tmp.groupby([by_field]).count()[[field]].reset_index()
    tmp.columns = [i for i in [by_field]] + ['count_of_' + field + '_by_' + str(by_field)]
    df = df.merge(tmp, on=by_field, how='left')
    return df[tmp.columns].drop([by_field], axis=1)


# distinct count agregate
def get_distinct_count(df, field, by_field):
    tmp = df[[by_field] + [field]].copy()
    tmp[field].fillna('xxx', inplace=True)
    tmp = tmp[[by_field] + [field]]
    tmp = tmp.drop_duplicates(inplace=False)
    tmp = tmp.groupby([by_field]).count()[[field]].reset_index()
    tmp.columns = [i for i in [by_field]] + ['distinct_count_of_' + field + '_by_' + str(by_field)]
    df = df.merge(tmp, on=by_field, how='left')
    return df[tmp.columns].drop([by_field], axis=1)


def tfidf(df, cols, transformers={}):
    dfs = []
    default = {
        'analyzer': 'word',
        'ngram_range': (1, 3),
        'stop_words': stop_words,
        'lowercase': True,
        'max_features': 50,
        'binary': True,
        'norm': None,
        'use_idf': True
    }
    for col in cols:

        if type(col) == str:
            col, conf = col, default
        else:
            col, conf = col[0], {**default, **col[1]}

        c = df[col].fillna('')
        if col not in transformers:
            transformers[col] = TfidfVectorizer(**conf).fit(c)
        vector = transformers[col].transform(c)
        d = pd.DataFrame(
            vector.todense(),
            columns=['tfidf_' + col + '_' + c for c in transformers[col].get_feature_names()]
        )
        #         d[col+'_n'] = c.str.split().apply(len)
        #         d[col+'_n_disctinct'] = c.str.split().apply(set).apply(len)

        dfs.append(d)
    return pd.concat(dfs, axis=1)


def match_col(X, columns):
    missing = set(columns) - set(X.columns)

    for col in missing:
        X[col] = 0

    return X[columns]


def dummies(df, cols, references={}):
    dfs = []
    for col in cols:
        dummies = pd.get_dummies(df[col])
        if col not in references:
            references[col] = dummies.columns.values
        else:
            dummies = match_col(dummies, references[col])
        dfs.append(dummies)

    return pd.concat(dfs, axis=1)


def label_encoder(df, cols, label_encoders={}):
    dfs = []
    for col in cols:
        #         if col not in label_encoders:
        #             label_encoders[col] = preprocessing.LabelEncoder().fit(df[col].astype(str))
        #         dfs.append(pd.DataFrame(label_encoders[col].transform(df[col].astype(str)),columns=[col]))
        dfs.append(pd.DataFrame(df[col].astype('category').cat.codes, columns=[col]))
    return pd.concat(dfs, axis=1)