import pandas
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate
import scipy
import scipy.sparse as sparse
import numpy as np
from scipy.sparse import coo_matrix
import json
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
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

def classification(category,train, label, list):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    #nomalize the distance
    adColumn = []
    lengthColumn = []
    for temp in train:
        temp = drop_stop_words(temp)
        temp = stem(temp)
        review_array = temp.split()
        averageDistance = 0.0
        length = 0.0
        for token in review_array:
            token = lower_no_punct(token)[0]
            if token in model.wv.vocab:
                d_min = min(model.wv.distances(token, list))
                averageDistance += d_min
                length += 1
        averageDistance = averageDistance / length
        adColumn.append(averageDistance)

    data_array = sparse.hstack((X_train_tfidf, np.array(adColumn)[:,None]))

    print("------------------------------------")
    print(category, " classification result:")
    scoring = ['precision', 'recall', 'accuracy', 'f1']
    kf = sklearn.model_selection.KFold(n_splits=10)
    clf = LinearSVC(C=1, max_iter=10000, dual=True).fit(data_array, label)
    scores = cross_validate(clf, data_array, label, scoring=scoring, cv=kf, return_train_score=False,
                            return_estimator=True)
    f_score = 2 * scores['test_recall'].mean() * scores['test_precision'].mean() / (
            scores['test_recall'].mean() + scores['test_precision'].mean())
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std()))
    print("Precision: %0.2f (+/- %0.2f)" % (scores['test_precision'].mean(), scores['test_precision'].std()))
    print("Recall: %0.2f (+/- %0.2f)" % (scores['test_recall'].mean(), scores['test_recall'].std()))
    print("F1-score: %0.2f (+/- %0.2f)" % (scores['test_f1'].mean(), scores['test_f1'].std()))
    index = 0
    if (category == "food"):
        print("enter")
        f = open(r"D:\graduate\textAnalyze\yelp\TFIDF_Unigrams_with_Embedding_Distance_Aggregate_for_food.json", 'w', encoding='UTF-8')
        for train_index, test_index in kf.split(data_array):
            for i in test_index:
                result = scores['estimator'][index].predict(data_array)
                json.dump({"text": str(train[i]), "actual label": int(label[i]), "predict label": int(result[i])}, f)
                f.write('\n')
            index += 1
    else:
        f = open(r"D:\graduate\textAnalyze\yelp\TFIDF_Unigrams_with_Embedding_Distance_Aggregate_for_service.json", 'w', encoding='UTF-8')

        for train_index, test_index in kf.split(data_array):
            for i in test_index:
                result = scores['estimator'][index].predict(data_array)
                json.dump({"text": str(train[i]), "actual label": int(label[i]), "predict label": int(result[i])}, f)
                f.write('\n')
            index += 1

#code from Abraham
def drop_stop_words(review):  # drops stop words from each review
    stop_words = set(stopwords.words('english'))
    stop_words.add("I")

    stop_words.remove("not")  # not may be important for analysis
    reviews_no_stop = []

        ## recover sentence
    tokens = review.split()
    review_no_stop_string = ""
    for token in tokens:
        token = token.lower()
        if token not in stop_words:
            if len(review_no_stop_string) == 0:
                review_no_stop_string = token
            elif len(token) > 1:
                review_no_stop_string = review_no_stop_string + " " + token
            else:
                review_no_stop_string = review_no_stop_string + token
    return review_no_stop_string

def stem(review):  # stems each review
    ps = PorterStemmer()
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
    return stemmed_sentence


def drop_punct(review):  # unimplemented, drops punctiation from each review
    return lower_no_punct(review)

def lower_no_punct(review): # strips punctuation and lower cases all words in any review provided
    review = review.split()
    new_sentence = []
    for token in review:
            punct_strip = str.maketrans('', '', string.punctuation)
            token = token.translate(punct_strip)
            new_sentence.append(token.lower())
    return new_sentence

def get_service_words(embedding): #gets a list of service words to compare the tokens to above
    service_words = ["service","waiter","host", "attent","hospitable","brought", "mean","nice","slow", "wrong"]
    service_word_tuples =  embedding.predict_output_word(["service","waiter","host", "attent","nice","hospitable","brought", "mean","slow", "wrong"], 20)
    for t in service_word_tuples:
        service_words.append(t[0])
    return set(service_words)

def get_food_words(embedding): #gets a list of food words to compare the tokens to above
    food_words = ["dish","delicious", "gross","bland","sandwich","taco","eat", "food"]
    for t in embedding.predict_output_word(["dish","delicious", "gross","bland","taco","sandwich","eat","food"],50):
        food_words.append(t[0])
    return set(food_words)

#end
data = pandas.DataFrame()

with open(r"D:\graduate\textAnalyze\yelp\labeled_reviews.json","r", encoding= 'UTF-8') as source:
    for line in source:
        lines = source.readlines()
        json_lines = "[" + ",".join(lines) + "]"
        json_data = json.loads(json_lines)
dataFood = pandas.read_csv(r"D:\graduate\textAnalyze\yelp\balanced_food_training_data.csv")
dataService = pandas.read_csv(r"D:\graduate\textAnalyze\yelp\balanced_service_training_data.csv")
model = Word2Vec.load(r"D:\graduate\textAnalyze\yelp\food_service_embedding.model")
dataFood = shuffle(dataFood)
dataService = shuffle(dataService)

#handel the original text data
service_list = get_service_words(model)
food_list = get_food_words(model)

food_list = food_list.difference(service_list)
service_list = service_list.difference(food_list)


classification("food", dataFood['text'], dataFood['label'], food_list)
classification("service", dataService['text'], dataService['label'], service_list)