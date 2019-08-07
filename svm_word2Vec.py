
import pandas
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate
from scipy.sparse.csr import csr_matrix
import json
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from gensim.models import Word2Vec
import  sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
def handle_label_food(row):
    if row['label'] in ("f", "b"):
        return 0
    else:
        return 1

def handle_label_service(row):
    if row['label'] in ("s", "b"):
        return 0
    else:
        return 1

def classification(category,train, label, list):
    indptr = [0]
    indices = []
    data = []
    vocabulary = {}
    #nomalize the distance
    maxD = []
    minD = []
    #sum = []
    count = 0
    for temp in train:
        maxD.append(-1.0)
        minD.append(100000.0)
        #sum.append(0.0)
        temp = drop_stop_words(temp)
        temp = stem(temp)
        review_array = temp.split()
        for token in review_array:
            token = lower_no_punct(token)[0]
            if token in model.wv.vocab:
                d_min = min(model.wv.distances(token, list))
                #sum[count] += d_min;
                if d_min > maxD[count]:
                    maxD[count] = d_min
                if d_min < minD[count]:
                    minD[count] = d_min
        count += 1
    count = 0
    for temp in train:
        temp = drop_stop_words(temp)
        temp = stem(temp)
        review_array = temp.split()
        for token in review_array:
            token = lower_no_punct(token)[0]
            if token in model.wv.vocab:
                d_min = min(model.wv.distances(token, list))
                index = vocabulary.setdefault(token, len(vocabulary))
                indices.append(index)
                # if d_min == 0:
                #     print(0)
                #else:
                d_min = (d_min - minD[count]) / (maxD[count] - minD[count])
                #d_min = d_min / sum[count]
                data.append(1 - d_min)
        count += 1
        indptr.append(len(indices))
    data_array = csr_matrix((data, indices, indptr), dtype=float)

    print("------------------------------------")
    print(category, " classification result:")
    scoring = ['precision', 'recall', 'accuracy', 'f1']
    kf = sklearn.model_selection.KFold(n_splits = 10)
    clf = LinearSVC(C=1, max_iter=10000, dual=True).fit(data_array, label)
    scores = cross_validate(clf, data_array, label, scoring=scoring, cv=kf, return_train_score=False, return_estimator = True)
    f_score = 2 * scores['test_recall'].mean() * scores['test_precision'].mean() / (
                scores['test_recall'].mean() + scores['test_precision'].mean())
    print("Accuracy: %0.4f (+/- %0.2f)" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std()))
    print("Precision: %0.4f (+/- %0.2f)" % (scores['test_precision'].mean(), scores['test_precision'].std()))
    print("Recall: %0.4f (+/- %0.2f)" % (scores['test_recall'].mean(), scores['test_recall'].std()))
    print("F1-score: %0.4f (+/- %0.2f)" % (scores['test_f1'].mean(), scores['test_f1'].std()))
    index = 0
    # predicted = cross_val_predict(clf, data_array, label, cv=kf)
    #
    # fig, ax = plt.subplots()
    # ax.scatter(label, predicted, edgecolors=(0, 0, 0))
    # ax.plot([label.min(), label.max()], [label.min(), label.max()], 'k--', lw=4)
    # ax.set_xlabel('Measured')
    # ax.set_ylabel('Predicted')
    # plt.show()
    if(category == "food"):
        f = open(r"D:\graduate\textAnalyze\yelp\food_Distance_of_Words.json", 'w',encoding='UTF-8')
        for train_index,test_index in kf.split(data_array):
            for i in test_index:
                result = scores['estimator'][index].predict(data_array)
                json.dump({"text": str(train[i]), "actual label": int(label[i]), "predict label": int(result[i])},f)
                f.write('\n')
            index += 1
    else:
        f = open(r"D:\graduate\textAnalyze\yelp\service_Distance_of_Words.json", 'w',encoding='UTF-8')

        for train_index,test_index in kf.split(data_array):
            for i in test_index:
                result = scores['estimator'][index].predict(data_array)
                json.dump({"text": str(train[i]), "actual label": int(label[i]), "predict label": int(result[i])},f)
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
dataFood = pandas.read_csv(r"D:\graduate\textAnalyze\yelp\balanced_food_training_data.csv")
dataService = pandas.read_csv(r"D:\graduate\textAnalyze\yelp\balanced_service_training_data.csv")
model = Word2Vec.load(r"D:\graduate\textAnalyze\yelp\food_service_embedding.model")
dataFood = shuffle(dataFood, random_state = 1324324)
dataService = shuffle(dataService, random_state = 1324324)
#handel the original text data
service_list = get_service_words(model)
food_list = get_food_words(model)

food_list = food_list.difference(service_list)
service_list = service_list.difference(food_list)

classification("food", dataFood['text'], dataFood['label'], food_list)
classification("service", dataService['text'], dataService['label'], service_list)