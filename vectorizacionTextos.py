# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import dok_matrix
from nltk import downloader
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
downloader.download("stopwords")

stopwords_list = set(stopwords.words("spanish"))
stemmer = SnowballStemmer("spanish")

word_list = ["Esto", "es", "una", "prueba"]
lower_words = [str.lower(word) for word in word_list]
filtered_words = [word for word in lower_words if word not in stopwords_list]
stemmed_words = [stemmer.stem(word) for word in filtered_words]
print(stemmed_words)
