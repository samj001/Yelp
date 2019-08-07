#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 16:21:29 2018

@author: fraifeld-mba
"""

import pandas as pd 
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import genesis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
from nltk.corpus import wordnet as wn
from sklearn import decomposition
import string
from gensim.models import Word2Vec
#from glove import Corpus, Glove

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import zipfile

from sklearn import model_selection


def edit_service_labels(r):
    service_dict = {}
    for i in r.keys():
        if (r[i][1]=='f'):
            service_dict[i] = [r[i][0], 0]
        else :
            service_dict[i] = [r[i][0], 1]
    return service_dict

def edit_food_labels(r):
    food_dict = {}
    for i in r.keys():
        if (r[i][1]=='s'):
            food_dict[i] = [r[i][0], 0]
        else :
            food_dict[i] = [r[i][0], 1]
    return food_dict

