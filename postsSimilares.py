# -*- coding: utf-8 -*-

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
import unicodedata
from nltk import downloader
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
downloader.download("stopwords")
np.set_printoptions(threshold=np.nan)

def preprocess(documentos):
    # Tratamiento de datos básico + stopwords + stemming
    new_documentos = []
    stopwords_list = set(stopwords.words("spanish"))
    stemmer = SnowballStemmer("spanish")
    for d in range(len(documentos)):
        unaccented_text = ''.join(c for c in unicodedata.normalize('NFD', documentos[d]) if unicodedata.category(c) != 'Mn')
        lower_words = [str.lower(word) for word in unaccented_text.split(" ")]
        filtered_words = [word for word in lower_words if word not in stopwords_list]
        stemmed_words = [stemmer.stem(word) for word in filtered_words]
        new_documentos.append(" ".join(stemmed_words))
    return new_documentos

categories = ["comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware", "comp.windows.x", "sci.space"]
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=categories)
newsgroup_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=categories)
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)