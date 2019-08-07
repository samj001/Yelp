import json
import numpy as np
import pandas
import math
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate, train_test_split
from sklearn import metrics
from sklearn import svm
import  sklearn
from sklearn.utils import shuffle
def handle_label_food(row):
    if row['label'] in ("f", "b"):
        return 1
    else:
        return 0

def handle_label_service(row):
    if row['label'] in ("s", "b"):
        return 1
    else:
        return 0

def classification(category,train, label):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    print("------------------------------------")
    print(category, " classification result:")

    scoring = ['precision', 'recall', 'accuracy', 'f1']
    kf = sklearn.model_selection.KFold(n_splits=10)
    clf_bow = LinearSVC(C=1 , penalty='l2', max_iter= 10000, dual=False).fit(X_train_counts, label)
    #clf_bow = svm.SVC(C=1).fit(X_train_counts, label)
    scores_bow = cross_validate(clf_bow, X_train_counts, label, scoring = scoring, cv = kf, return_train_score=False, return_estimator = True)

    print("Result for bag of words: ")
    #print("Precision: %0.2f (+/- %0.2f)" % (scores_bow['accuracy'].mean(), scores_bow['accuracy'].std()))
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores_bow['test_accuracy'].mean(), scores_bow['test_accuracy'].std()))
    print("Precision: %0.2f (+/- %0.2f)" % (scores_bow['test_precision'].mean(), scores_bow['test_precision'].std()))
    print("Recall: %0.2f (+/- %0.2f)" % (scores_bow['test_recall'].mean(), scores_bow['test_recall'].std()))
    print("F1-score: %0.2f (+/- %0.2f)" % (scores_bow['test_f1'].mean(), scores_bow['test_f1'].std()))


    clf_tfidf = LinearSVC(C=1 , max_iter = 10000, dual=True).fit(X_train_tfidf, label)
    #clf_tfidf = svm.SVC(C=1).fit(X_train_counts, label)
    scores_tfidf = cross_validate(clf_tfidf,X_train_tfidf, label, scoring=scoring, cv = 10, return_train_score = False, return_estimator = True)
    f_score_tfidf = 2 * scores_tfidf['test_recall'].mean() * scores_tfidf['test_precision'].mean() / (scores_tfidf['test_recall'].mean() + scores_tfidf['test_precision'].mean())
    #print(clf_tfidf.predict(X_train_tfidf))
    print("Result for tf-idf: ")
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores_bow['test_accuracy'].mean(), scores_bow['test_accuracy'].std()))
    print("Precision: %0.2f (+/- %0.2f)" % (scores_tfidf['test_precision'].mean(), scores_tfidf['test_precision'].std()))
    print("Recall: %0.2f (+/- %0.2f)" % (scores_tfidf['test_recall'].mean(), scores_tfidf['test_recall'].std()))
    print("F1-score: %0.2f (+/- %0.2f)" % (scores_tfidf['test_f1'].mean(), scores_tfidf['test_f1'].std()))

    index = 0
    if (category == "food"):
        fb = open(r"D:\graduate\textAnalyze\yelp\food_BoW.json", 'w', encoding='UTF-8')
        for train_index, test_index in kf.split(X_train_counts):
            for i in test_index:
                result = scores_bow['estimator'][index].predict(X_train_counts)
                json.dump({"text": str(train[i]), "actual label": int(label[i]), "predict label": int(result[i])}, fb)
                fb.write('\n')
            index += 1

        ft = open(r"D:\graduate\textAnalyze\yelp\food_tf_idf.json", 'w', encoding='UTF-8')
        index = 0
        for train_index, test_index in kf.split(X_train_tfidf):
            for i in test_index:
                result = scores_tfidf['estimator'][index].predict(X_train_tfidf)
                json.dump({"text": str(train[i]), "actual label": int(label[i]), "predict label": int(result[i])}, ft)
                ft.write('\n')
            index += 1
    else:
        fb = open(r"D:\graduate\textAnalyze\yelp\service_BoW.json", 'w', encoding='UTF-8')
        for train_index, test_index in kf.split(X_train_counts):
            for i in test_index:
                result = scores_bow['estimator'][index].predict(X_train_counts)
                json.dump({"text": str(train[i]), "actual label": int(label[i]), "predict label": int(result[i])}, fb)
                fb.write('\n')
            index += 1

        ft = open(r"D:\graduate\textAnalyze\yelp\service_tf_idf.json", 'w', encoding='UTF-8')
        index = 0
        for train_index, test_index in kf.split(X_train_tfidf):
            for i in test_index:
                result = scores_tfidf['estimator'][index].predict(X_train_tfidf)
                json.dump({"text": str(train[i]), "actual label": int(label[i]), "predict label": int(result[i])}, ft)
                ft.write('\n')
            index += 1



dataFood = pandas.read_csv(r"D:\graduate\textAnalyze\yelp\balanced_food_training_data.csv")
dataService = pandas.read_csv(r"D:\graduate\textAnalyze\yelp\balanced_service_training_data.csv")
dataFood = shuffle(dataFood)
dataService = shuffle(dataService)

classification("food", dataFood['text'], dataFood['label'])
classification("service", dataService['text'], dataService['label'])