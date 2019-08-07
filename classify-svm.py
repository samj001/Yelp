#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 12:25:56 2018

@author: fraifeld-mba
"""
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import cross_val_score

def merge_labeled_reviews(i=0, k=100):
    
    reviews = {}
    with open("labeled_reviews_1to200.json","r") as f1:
        with open("labeled_reviews.json", "r") as f2:
            for f in [f1, f2]:
                count = 0
                for line in f:
                    if count < i:
                        pass
                    elif count < k+1:
                        l = json.loads(line)
                        r_id = l["review_id"]
                        if (r_id not in reviews.keys() ):
                            reviews[r_id] = [l["text"], l["label"]]
                        elif (reviews[r_id][1]==l["label"]):
                            pass
                        elif (reviews[r_id][1] != l["label"]):
                            if (reviews[r_id][1] == "b" or l["label"] == "b"):
                                reviews[r_id] = [l["text"], "b"]
                    else:
                        break
                    count = count + 1
    return reviews


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


def drop_stop_words(reviews): # drops stop words
    stop_words = set(stopwords.words('english')) 
    stop_words.add("I")

    stop_words.remove("not") # not may be important for analysis
    reviews_no_stop = []
    for review in reviews:
        
        ## recover sentence
        tokens = review.split()
        review_no_stop_string = ""
        for token in tokens:
            token = token.lower()
            if token not in stop_words:
                if len(review_no_stop_string) == 0:
                    review_no_stop_string = token
                elif len(token) > 1 :
                    review_no_stop_string = review_no_stop_string + " " + token
                else:
                    review_no_stop_string = review_no_stop_string + token
        reviews_no_stop.append(review_no_stop_string)
    return reviews_no_stop

def stem(reviews):
    ps = PorterStemmer()
    reviews_stemmed = []

    for review in reviews:
        tokens = review.split()
        stemmed_sentence = ""
        for token in tokens:
            token = ps.stem(token)
            if len(stemmed_sentence) == 0:
                stemmed_sentence = token
            elif len(token) > 1:
                stemmed_sentence = stemmed_sentence + " " + token
            else:
                stemmed_sentence = stemmed_sentence + token
        reviews_stemmed.append(stemmed_sentence)
    return reviews_stemmed

def bag_words(reviews):
    reviews = drop_stop_words(reviews)
    reviews = stem(reviews)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(reviews)
    X = X.toarray()   
    return X

def weight_by_tfidf(reviews):
    reviews = drop_stop_words(reviews)
    reviews = stem(reviews)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(reviews)
    X = X.toarray()   
    return X                    
 

def get_Y(r):
    labels = []
    vals = list(r.values())
    for i in vals:
        labels.append(i[1])
    return labels

def get_X(r, method = "B"):
    reviews = []
    vals = list(r.values())
    for i in vals:
        reviews.append(i[0])
    if method == "B":
        return bag_words(reviews)
    else:
        return weight_by_tfidf(reviews)

def k_fold(X, Y, k = 10):
    X = np.normalize(X)
    model = svm.SVC()
    model.fit(X, Y)
    scoring_param = ['accuracy', 'precision', 'recall', 'f1']
    final_scores = {}
    for s in scoring_param:
        scores = cross_val_score(model, X, Y, cv=k, scoring = s)
        final_scores[s] = [np.mean(scores), np.std(scores)] 
    return final_scores

if __name__ == "__main__":
    print("\n")
    r = merge_labeled_reviews(0,300)
    s_dict = edit_service_labels(r)
    f_dict = edit_food_labels(r)
    

    # Food -- X
    tfidf_food = get_X(f_dict, method = "T")
    bow_food = get_X(f_dict, method = "B")
    
    # Food -- Y
    food_labels = get_Y(f_dict)
    
    # Service -- X
    
    tfidf_service = get_X(s_dict, method = "T")
    bow_service = get_X(s_dict, method = "B")
    
    # Service -- Y
    
    service_labels = get_Y(s_dict)

        
    # Get Results
    
    s_t = k_fold(tfidf_service, service_labels)
    s_b = k_fold(bow_service, service_labels)

    f_t = k_fold(tfidf_food, food_labels)
    f_b = k_fold(bow_food, food_labels)
    
    
    # Print results
    
    results = [f_t, f_b, s_t, s_b]
    X = ['tfidf-food','bow-food', 'tfidf-service', 'bow-service']
    scores = ['accuracy', 'precision', 'recall', 'f1']
    
    for r in enumerate(results):
       print(X[r[0]])
       for s in scores:
           print(s + "  =  mean: " + str(r[1][s][0]) + " sigma: " + str(r[1][s][1]) )
        


    
    
    


