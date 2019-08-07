
import numpy as np
import scipy.sparse as sparse
import json
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import genesis
from nltk.corpus import wordnet as wn
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn import metrics
from sklearn.utils import shuffle


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

def balance(label,text):
    new_label = []
    new_text = []
    count1 = 0
    count0 = 0
    for i in label:
        if i == 1:
            count1 += 1
        else:
            count0 += 1
    
    if count0 == count1:
        return (label,text)
    elif count0 < count1:
        count = 0
        for n in range(len(label)):
            if label[n] == 0:
                new_label.append(label[n])
                new_text.append(text[n])
            else:
                count += 1
                if count <= count0:
                    new_label.append(label[n])
                    new_text.append(text[n])

    else:
        count = 0
        for n in range(len(label)):
            if label[n] == 1:
                new_label.append(label[n])
                new_text.append(text[n])
            else:
                count += 1
                if count <= count1:
                    new_label.append(label[n])
                    new_text.append(text[n])

    return(new_label,new_text)


def shuffle_data(label, reviews):
    dataset = []
    for i, review in enumerate(reviews):
        dataset.append((review,label[i]))

    dataset = shuffle(dataset)
    
    return_review = []
    return_label = []

    for (n, item) in enumerate(dataset):
        return_review.append(item[0]) 
        return_label.append(item[1])

    return (return_label, return_review)



# input: single review string, return: string
def pre_process(review): 
    review = drop_stop_words(review)
    review = lower_no_punct(review)
    review = stem(review)
    ###review_list = review.split()
    return review


# returns each review represented by unigram tfidfs
def weight_by_tfidf(reviews, ngram_range): 
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    X = vectorizer.fit_transform(reviews)
    X = X.toarray()   
    return X 



def feature_vector_four(reviews, embedding, t= "TFIDF"):
    food_serv_vec = food_serv_sim_vec_embedding(reviews, embedding)  # TFIDF, wordnet aggregate
    # breaks wordnet
    clean_reviews = []
    for review in reviews:
        review = pre_process(review)
        clean_reviews.append(review)
    
    if t == "TFIDF":
        m =  weight_by_tfidf(clean_reviews,(1,1))
    else:
        m = bag_words(clean_reviews)
    return_array = []
    for i, review in enumerate(clean_reviews):
        return_array.append(np.append(m[i],[food_serv_vec[i][0],food_serv_vec[i][1]]))
    return return_array



def food_serv_sim_vec_embedding(reviews, embedding):
    
    ## similar to the wup aggregation function above
    # compares each token to service words/food words and returns an aggregated vector
    
    #words = list(embedding.wv.vocab)
    dropped_reviews = []
    for review in reviews:
        drop = drop_stop_words(review)
        drop = stem(drop)
        dropped_reviews.append(drop)

    reviews = dropped_reviews
    
    service_list = get_service_words(embedding)
    food_list = get_food_words(embedding)
    
    food_list = food_list.difference(service_list)
    service_list = service_list.difference(food_list)
    
    return_vec = []
    for review in reviews:
        food_d = []
        service_d = []
        review_array = review.split()
        for token in review_array:
            token = lower_no_punct(token)[0]
            if token in embedding.wv.vocab:
                food_min = min(embedding.wv.distances(token, food_list))
                service_min = min(embedding.wv.distances(token, service_list))
                food_d.append(food_min)
                service_d.append(service_min)
                    #print(token + " FOOD : " + str(food_min) + " SERVICE: " + str(service_min))

        return_vec.append([np.mean(food_d), np.mean(service_d)])
     
    return return_vec


def classification (category_name, train, label):

    y = label
    X_train = feature_vector_four(train, embedding)

    print("------------------------------------")
    print(category_name, " classification result:")

    scoring = ['precision', 'recall', 'accuracy', 'f1']
    kf = KFold(n_splits=10)
    clf = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr').fit(X_train, y)
    scores = cross_validate(clf, X_train, y, scoring = scoring, cv=kf, return_train_score=False)
    f_score = 2 * scores['test_recall'].mean() * scores['test_precision'].mean() / (scores['test_recall'].mean() + scores['test_precision'].mean())

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std()))
    print("Precision: %0.2f (+/- %0.2f)" % (scores['test_precision'].mean(), scores['test_precision'].std()))
    print("Recall: %0.2f (+/- %0.2f)" % (scores['test_recall'].mean(), scores['test_recall'].std()))
    print("F1-score: %0.2f (+/- %0.2f)" % (scores['test_f1'].mean(), scores['test_f1'].std()))



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
    new_sentence = ''
    for token in review:
            punct_strip = str.maketrans('', '', string.punctuation)
            token = token.translate(punct_strip)
            new_sentence = new_sentence + token.lower() + ' '
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

# end

food_label, food_train = balance(food_label,food_train)
food_label, food_train = shuffle_data(food_label,food_train)
service_label, service_train = balance(service_label,service_train)
service_label, service_train = shuffle_data(service_label, service_train)

embedding = Word2Vec.load("food_service_embedding.model")

classification("food", food_train,food_label)
classification("service", service_train,service_label)













