import re
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer


def custom_preprocessor(text):
    text = text.lower()
    text = re.sub("\[.*?\]", '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub("https?://\S+|www\.\S+", "", text)
    text = re.sub("<.*?>+", "", text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub("\n", '', text)
    text = re.sub('\w*\d\w', '', text)

    return text


def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    sw = stopwords.words('english')
    count_vectorizer = CountVectorizer(token_pattern=r'\w{1,}', ngram_range=(1, 2),
                                       stop_words=sw, preprocessor=custom_preprocessor)
    count_vectorizer.fit(train["text"])

    train_vectors = count_vectorizer.fit_transform(train["text"])
    test_vectors = count_vectorizer.fit_transform(test["text"])

    clf = LogisticRegression(C=1.0)
    scores = cross_val_score(clf, train_vectors, train["target"], cv=5, scoring="f1")
    print(scores)


if __name__ == '__main__':
    main()
