
import numpy as np
import json
import math
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn import metrics

f = open("labeled_reviews.json", encoding='UTF-8')
data_food = {}
data_service = {}
food_label = []
food_train = [] 
service_label = [] 
service_train = []

for line in f:
    B = json.loads(line)
    data_food["review_id"] = B["review_id"]
    data_service["review_id"] = B["review_id"]
    data_food["text"] = B["text"]
    data_service["text"] = B["text"]
    food_train.append(B["text"])
    service_train.append(B["text"])

    if B["label"] == "b":
    	data_food["label"] = 1
    	food_label.append(1)
    	data_service["label"] = 1
    	service_label.append(1)

    elif B["label"] == "n":
    	data_food["label"] = 0
    	food_label.append(0)
    	data_service["label"] = 0
    	service_label.append(0)

    elif B["label"] == "f":
    	data_food["label"] = 1
    	food_label.append(1)
    	data_service["label"] = 0
    	service_label.append(0)

    else:
    	data_food["label"] = 0
    	food_label.append(0)
    	data_service["label"] = 1
    	service_label.append(1)
f.close()

def classification(category,train,label,test_percent):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    y = label

    print("------------------------------------")
    print(category, " classification result:")

    scoring = ['precision', 'recall']
    clf_bow = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr').fit(X_train_counts, y)
    scores_bow = cross_validate(clf_bow, X_train_counts, y, scoring = scoring, cv=10, return_train_score=False)

    clf_tfidf = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr').fit(X_train_tfidf, y)
    scores_tfidf = cross_validate(clf_tfidf, X_train_tfidf, y, scoring = scoring,cv=10, return_train_score=False)

    print("Bag of words result: ")
    print("Precision: %0.2f (+/- %0.2f)" % (scores_bow['test_precision'].mean(), scores_bow['test_precision'].std()*2))
    print("Recall: %0.2f (+/- %0.2f)" % (scores_bow['test_recall'].mean(), scores_bow['test_recall'].std()*2))
    f_score_bow = 2 * scores_bow['test_recall'].mean() * scores_bow['test_precision'].mean() / (scores_bow['test_recall'].mean() + scores_bow['test_precision'].mean())
    print("F1-score: %0.2f" % (f_score_bow), "\n")
    
    print("tf_idf result : ")
    print("Precision: %0.2f (+/- %0.2f)" % (scores_tfidf['test_precision'].mean(), scores_tfidf['test_precision'].std()*2))
    print("Recall: %0.2f (+/- %0.2f)" % (scores_tfidf['test_recall'].mean(), scores_tfidf['test_recall'].std()*2))
    f_score_tfidf = 2 * scores_tfidf['test_recall'].mean() * scores_tfidf['test_precision'].mean() / (scores_tfidf['test_recall'].mean() + scores_tfidf['test_precision'].mean())
    print("F1-score: %0.2f" % (f_score_tfidf), "\n")
    

classification("food", food_train,food_label,0.2)
classification("service", service_train,service_label,0.2)
