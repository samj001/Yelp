#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 12:25:56 2018

@author: fraifeld-mba
"""
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

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


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 17:00:32 2018

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

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


# This funciton is a helper. It reads in a review at a time
def parse_json(line):
    j = json.loads(line)
    return j

def load_data(i=1, k=150000): ## loads data indexed from i to k.  Defaults to whole set
    df = pd.DataFrame()
    with open("merge2.json","r") as f:
        count = 1
        for line in f:
            if count < i:
                pass
            elif count < k+1:
                df_plus = pd.read_json(line, lines = True)
                df = df.append(df_plus)
                
            else:
                break
            count = count + 1
    return df.reset_index()

def drop_stop_words(reviews): # drops stop words from each review
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

def stem(reviews): # stems each review
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

def drop_punct(reviews): # unimplemented, drops punctiation from each review
    reviews_no_punct = []
    for review in reviews:
        reviews_no_punct.append(" ".join(lower_no_punct(review)))
    return reviews_no_punct


def load_review_text(i,k): # efficient read of review text into an array
    reviews = []
    with open("merge2.json","r") as f:
        count = 1
        for line in f:
            if count < i:
                pass
            elif count < k+1:
                reviews.append(json.loads(line)["text"])
                
            else:
                break
            count = count + 1
    return reviews



def lower_no_punct(review): # strips punctuation and lower cases all words in any review provided
    review = review.split()
    new_sentence = []
    for token in review:
            punct_strip = str.maketrans('', '', string.punctuation)
            token = token.translate(punct_strip)
            new_sentence.append(token.lower())
    return  new_sentence


def bag_words(reviews): # returns a BoW representation of all reviews provided
    reviews = drop_stop_words(reviews)
    reviews = stem(reviews)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(reviews)
    X = X.toarray()   
    return X


def get_unigram_frequencies(reviews): # returns unigram frequencies after dropping stop words
    reviews = drop_stop_words(reviews)
    reviews = stem(reviews)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(reviews)
    return (list(zip(vectorizer.get_feature_names(), np.ravel(X.sum(axis=0)))))


def print_ith_freq_unigram(reviews, i=1): # prints the i'th most frequent unigram after dropping stop words and stemming
    x = get_unigram_frequencies(reviews)
    print(sorted(x,key=lambda item:item[1])[-i])
    

def bigrams(reviews): # returns a bigram representation of the reviews
    vectorizer = CountVectorizer(ngram_range = (1,2), token_pattern = r'\b\w+\b', min_df=1)
    X = vectorizer.fit_transform(reviews)
    X = X.toarray()   
    return X

def get_bigram_frequencies(reviews): # gets the bigram frequencies
     vectorizer = CountVectorizer(ngram_range = (1,2), token_pattern = r'\b\w+\b', min_df=1)
     X = vectorizer.fit_transform(reviews)
     return (list(zip(vectorizer.get_feature_names(), np.ravel(X.sum(axis=0)))))

def weight_by_tfidf(reviews, ngram_range=(1,1)): # returns each review represented by unigram tfidfs
    vectorizer = TfidfVectorizer(ngram_range=ngram_range)
    X = vectorizer.fit_transform(reviews)
    X = X.toarray()   
    return X                    

def get_tfids(reviews): # returns a list of tfidfs
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(reviews)
    return (list(zip(vectorizer.get_feature_names(), np.ravel(X.mean(axis=0)))))
    
def print_ith_highest_tfidf(reviews, i=1): # prints ith highest tfidf
    x = get_tfids(reviews)
    print(sorted(x,key=lambda item:item[1])[-i])
    

 
def tag_hypernyms(reviews): # gets wordnet hypernyms for each token and returns a representation of them
    reviews = drop_stop_words(reviews)
    full_list_of_hypernums = []
    for review in reviews:
        tokens = review.split()
        review_hypernyms = []
        for token in tokens:
            punct_strip = str.maketrans('', '', string.punctuation)
            token = token.translate(punct_strip)
            token_syns = wn.synsets(token.lower())
            hypernyms = []
            if len(token_syns) > 0:
                hypernyms = token_syns[0].hypernyms() # This needs to be justifed
            if len(hypernyms) > 0:
                if "_" in hypernyms[0].lemma_names()[0]:
                    hnym = hypernyms[0].lemma_names()[0].split("_")
                    review_hypernyms.append(hnym[len(hnym)-1])

                else:
                    review_hypernyms.append(hypernyms[0].lemma_names()[0])
        full_list_of_hypernums.append(review_hypernyms)
    return full_list_of_hypernums


def food_serv_sim_vec_wn(reviews, picky=False, test = False, vtest = False): 
    # calculates the WUP score of each token. 
    #Conditionally appends it to a vector, then returns a list of vectors 
    # that represent an aggregated wup score [food,service] for each review
    #Only preprocessing allowed:
    #reviews = drop_punct(reviews)
    #reviews = drop_stop_words(reviews) # drop the stop words as a preprocessing step
    
    # extra preprocessing breaks the algorithm because it leads to inconsistency with wordnet.
    food_words = wn.synsets("food") # as in "food or drink"
    service_words = [wn.synsets("waiter")[0],wn.synsets("service")[14]] # this version of service is what we meant
    final_list = [] # this will be returned
    for review in reviews:
        review = review.split() # tokenize each review
        noun_count = 0
        food_scores = []
        service_scores = []
        for word in review:
            #preprocess the token

            
            # Crete arrays to fill with similarity values
            # we do this because we will have lots of  words in the synset, and we will grab the 
            #max similarity value for each of these
            similarities_food = []
            similarities_serv = []


            
            syns = wn.synsets(word, pos="n") # we only want to look at nouns because 
            # a) they are most informative and b) wup trips up on other parts of speech
            
            
            if len(syns) > 0:
                noun_count = noun_count + 1

                for w1 in syns:
                    similarities_food.append(max( w1.wup_similarity(food_words[0]),
                                                 w1.wup_similarity(food_words[1]))) 
                    similarities_serv.append(max(w1.wup_similarity(service_words[0]), 
                                                 w1.wup_similarity(service_words[1])))
                
                
                sim_food = max(similarities_food)
                sim_serv = max(similarities_serv)
                
                ## append the similarities to an array. This array represents the metric for each review
                if picky:
                    if sim_food - sim_serv > .3 or sim_food > .6:
                        food_scores.append(sim_food)
                        if(test):
                            print(word + "s: " + str(sim_serv) + " , f: " + str(sim_food))
                    if sim_serv - sim_food > .3:
                        service_scores.append(sim_serv)
                        if(test):
                            print(word + "s: " + str(sim_serv) + " , f: " + str(sim_food))
                else:
                    food_scores.append(sim_food)
                    service_scores.append(sim_serv)


        if noun_count: 
            final_list.append([sum(food_scores)/noun_count, sum(service_scores)/noun_count])
        else:
            final_list.append([sum(food_scores), sum(service_scores)])

    return final_list
    

def get_embedding(reviews):
    reviews = drop_stop_words(reviews)
    reviews = stem(reviews)
    
    sentences = []
    for review in reviews:
        sentences.append(lower_no_punct(review))
   
    # Train the embedding
    
    embedding = Word2Vec(sentences, min_count=1)
    return embedding

def tag_word_embedding_pca(reviews, embedding ,return_val = "mean"):

    # Perform PCA of the embedding - this will be the feature sent to the array for classification

    words = list(embedding.wv.vocab)
    X = embedding[embedding.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    embedding_dict = {}
    for i, word in enumerate(words):
        embedding_dict[word] = [result[i, 0], result[i, 1]]
    
    
    # return the mean of the dimensions across the words in the sentences 
    # consider returning something more informative - maybe turn this into a vectorizer
    embedding_tags = []
    for review in reviews:
        review_embeddings = []
        for token in review.split():
            token = lower_no_punct(token)[0]
            review_embeddings.append(embedding_dict[token])
        if return_val == "mean":
            embedding_tags.append([np.mean(review_embeddings[0]), np.mean(review_embeddings[1])])
    return embedding_tags

def food_serv_sim_vec_embedding(reviews, embedding):
    
    ## similar to the wup aggregation function above
    # compares each token to service words/food words and returns an aggregated vector
    
    #words = list(embedding.wv.vocab)
    reviews = drop_stop_words(reviews)
    reviews = stem(reviews)
    
    
    
    
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
            
        

def get_service_words(embedding): #gets a list of service words to compare the tokens to above
    service_words = ["service","waiter","wait","time","host","staff", "attent","hospitable","brought", "mean","nice","slow", "wrong"]
    service_word_tuples =  embedding.predict_output_word(service_words, 30)
    for t in service_word_tuples:
        service_words.append(t[0])
    return set(service_words)

def get_food_words(embedding): #gets a list of food words to compare the tokens to above
    food_words = ["dish","dessert","chocol","cheese","delicious", "gross","bland","bagel","sandwich","taco","naan","chinese","eat", "food"]
    for t in embedding.predict_output_word(food_words,50):
        food_words.append(t[0])
    return set(food_words)

def tag_doc_embedding(): # use doc2vec
    pass
    
    
def plot_embedding(embedding, wordlist = []): # plots the word embedding on an x,y plane
    if len(wordlist) == 0:
        wordlist = list(embedding.wv.vocab)
    X = embedding[embedding.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    pyplot.scatter(result[:, 0], result[:, 1])
    words = list(embedding.wv.vocab)
    for i, word in enumerate(words):
        if word in wordlist:
            plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    
    pyplot.show()


def fast_sim(token, embedding, comparison):  
     syns = wn.synsets(token, pos="n") # we only want to look at nouns because 
            # a) they are most informative and b) wup trips up on other parts of speech
     if len(syns) > 0:

         similarities = []
         for w1 in syns:
             for c in comparison:
                 comp_sim = []
                 comp_sim.append(w1.wup_similarity(c))
             similarities.append(max(comp_sim))
         return max(similarities)
     
     else:
         return 0

def fast_comparison_embedding(token, embedding, comparison):
    return min(embedding.wv.distances(token, comparison))

def get_vocab(reviews):
    reviews = drop_stop_words(reviews)
    reviews = stem(reviews)
    reviews = drop_punct(reviews)
    vocab = []
    for review in reviews:
        review_tokens = review.split()
        for t in review_tokens:
            vocab.append(t)
                
    return vocab


def twenty_most_similar(reviews, category):
    if category == "food":
        comparison = wn.synsets("food")
    elif category == "service":
        comparison = [wn.synsets("waiter")[0],wn.synsets("service")[14]]
    
    return_array = []
    
    for i, review in enumerate(reviews):
          review = review.split()
          most_similar = []
          for token in review:
              try:
                  similarity = fast_sim(token, embedding, comparison)
              except:
                  similarity = 0
              
              # consider adding it to most_similar
              if len(most_similar) < 20 or similarity > min(most_similar):
                  most_similar.append(similarity)
                  
          while len(most_similar) < 20:
              most_similar.append(np.mean(most_similar))
          
          return_array.append(sorted(most_similar, reverse=True)[0:20])
          
    return return_array
        

def twenty_closest(reviews, category, embedding):

    if category == "food":
        comparison = get_food_words(embedding)
    elif category == "service":
        comparison = get_service_words(embedding)
    
    return_array = []
    
    for i, review in enumerate(reviews):
          review = review.split()
          closest = []
          for token in review:
              if token in embedding.wv.vocab:
            #  try:
                  distance = fast_comparison_embedding(token, embedding, comparison)
              else:
                  distance = 1
              
              # consider adding it to closest
              if len(closest) < 20 or distance < max(closest):
                  closest.append(distance)
                  
          while len(closest) < 20:
              
              closest.append(np.mean(closest))
          
          return_array.append(sorted(closest)[0:20])
          
    return return_array
        
              
def test_arr(reviews): ## Prints the array of features for testing purposes
    reviews = drop_stop_words(reviews)
    reviews = stem(reviews)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(reviews)
    bow = X.toarray()    
    for i in range(len(bow)):
        for j in range(len(bow[i])):
            if bow[i][j] == 1:
                print(vectorizer.get_feature_names()[j])

def get_review_length(reviews):
    lenghts = []
    for review in reviews:
        lenghts.append(len(review.split()))
    return lenghts
    
def working_space(reviews): ## JUST RANDOM PLACE TO MESS AROUND
    
    
    ### TFIDF set up
    big_array = weight_by_tfidf(reviews)
    w_norm = normalize(big_array)
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(big_array)
    y_km = kmeans.fit_predict(w_norm)
    pca2D = decomposition.PCA(2)
    plot_columns = pca2D.fit_transform(w_norm)
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=y_km) 


    # WN
    h = food_serv_sim_vec_wn(reviews)
    dt=np.dtype('float','float') 
    h_1 = np.array(h,dtype=dt)
    norm_h = normalize(h_1)
    final_big_array = []
    
    for i in range(len(big_array)):
         x = np.append(big_array[i], [norm_h[i,0], norm_h[i,1]])
         if i == 0:
            final_big_array = x
         else:
            final_big_array = np.vstack([final_big_array, x])
    
    f_norm = normalize(final_big_array)
    kmeans = KMeans(n_clusters=2)

    kmeans.fit(final_big_array)
    y_km = kmeans.fit_predict(f_norm)

    pca2D = decomposition.PCA(2)
    plot_columns = pca2D.fit_transform(f_norm)
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=y_km) 
    pylab.savefig("wn-kmeans.png")


    # Embedding
    h = food_serv_sim_vec_embedding(reviews, embedding)
    dt=np.dtype('float','float') 
    h_1 = np.array(h,dtype=dt) 
    h_1 = np.nan_to_num(h_1)
    norm_h = normalize(h_1)

    final_big_array = []
    
    for i in range(len(big_array)):
         x = np.append(big_array[i], [norm_h[i,0], norm_h[i,1]])
         if i == 0:
            final_big_array = x
         else:
            final_big_array = np.vstack([final_big_array, x])
    
    f_norm = normalize(final_big_array)
    kmeans = KMeans(n_clusters=2)

    kmeans.fit(final_big_array)
    y_km = kmeans.fit_predict(f_norm)

    pca2D = decomposition.PCA(2)
    plot_columns = pca2D.fit_transform(final_big_array)
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=y_km) 
    pylab.savefig("embed-kmeans.png")

      # Granular
    h = bag_words_weight_embedding(reviews,embedding)
    dt=np.dtype('float','float') 
    norm_h = normalize(h)


    kmeans = KMeans(n_clusters=10)

    kmeans.fit(h)
    y_km = kmeans.fit_predict(norm_h)

    pca2D = decomposition.PCA(2)
    plot_columns = pca2D.fit_transform(norm_h)
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=y_km) 
    pylab.savefig("embed-kmeans.png")
    
    
    
    
    
    
    
    ### WN VECTOR SET UP
    
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(h_1)
    y_km = kmeans.fit_predict(norm_h)
    plt.scatter(x=h_1[:,0], y=h_1[:,1], c=y_km) 




    import pylab
    pylab.savefig("tfidf-kmeans")
    
    ### Embed Setup
    h = food_serv_sim_vec_embedding(reviews, embedding)
    dt=np.dtype('float','float') 
    h_1 = np.array(h,dtype=dt)
    norm_h = normalize(h_1)
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(h_1)
    y_km = kmeans.fit_predict(norm_h)
    plt.scatter(x=h_1[:,0], y=h_1[:,1], c=y_km) 
    import pylab
    pylab.savefig("embed-kmeans")
    
    

    
    kmeans.fit(final_big_array)
    y_km = kmeans.fit_predict(f_norm)
    
    
    plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=y_km)
    pylab.savefig("wn-kmeans")
    

    for i in range(370,500):
        x = food_serv_sim_vec_wn([reviews[i]])
        if (x[0][0] > x[0][1]):
            print(reviews[i])
            print(x)
            break
        


def pre_process(reviews):
    reviews = drop_stop_words(reviews)
    reviews = stem(reviews)
    reviews = drop_punct(reviews)

    return reviews

def pre_process_wn(reviews): 
    reviews = drop_punct(reviews)
    reviews = drop_stop_words(reviews) # drop the stop words as a preprocessing step
    return reviews
        
def feature_vector_one(reviews): #TFIDF Unigrams
    reviews = pre_process(reviews)
    tfidf =  weight_by_tfidf(reviews)
    return_array = []
    for i, review in enumerate(reviews):
        return_array.append(tfidf[i])
    return return_array

def feature_vector_two(reviews): #TFIDF Unigrams/Bigrams
    reviews = pre_process(reviews)
    tfidf =  weight_by_tfidf(reviews,(1,2))
    return_array = []
    for i, review in enumerate(reviews):
        return_array.append(tfidf[i])
    return return_array


##  the hypothesis behind 3-4 is that service reviews have more service words than food words and vice versa
## that intuition is not behind 5-6, which just claim that food reviews have food words and service reviews have service words
    

def feature_vector_three(reviews, t= "TFIDF"):
    food_serv_vec = food_serv_sim_vec_wn(reviews, picky=True) # Bow or TFIDF, wordnet aggregate
    # breaks wordnet
    reviews = pre_process(reviews)
    
    if t == "TFIDF":
        m =  weight_by_tfidf(reviews)
    else:
        m = bag_words(reviews)
    return_array = []
    for i, review in enumerate(reviews):
        return_array.append(np.append(m[i],[food_serv_vec[i][0],food_serv_vec[i][1]]))
    return return_array


def feature_vector_four(reviews, embedding, t="TFIDF"):
    food_serv_vec = food_serv_sim_vec_embedding(reviews, embedding) # TFIDF, embedding aggregate
    # breaks wordnet
    reviews = pre_process(reviews)
    if t == "TFIDF":
        m =  weight_by_tfidf(reviews)
    else:
        m = bag_words(reviews)
    return_array = []
    for i, review in enumerate(reviews):
        return_array.append(np.append(m[i],[food_serv_vec[i][0],food_serv_vec[i][1]]))
    return return_array



def feature_vector_five(reviews, category, t="TFIDF"): #Bow or TFIFD, Worndet - 20 most similar
    reviews_wn = pre_process_wn(reviews)
    tsm = twenty_most_similar(reviews_wn, category)
    
    reviews = pre_process(reviews)
    if t == "TFIDF":
        m =  weight_by_tfidf(reviews)
    else:
        m = bag_words(reviews)
    return_array = []
    for i, review in enumerate(reviews):
        concat_array = tsm[i]
        #print(concat_array)
        return_array.append(np.concatenate([m[i],concat_array]))
    return return_array
    

def feature_vector_six(reviews, category, embedding): #Bow Unigrams/Bigrams, Embedding - 20 closest

    reviews = pre_process(reviews)
    tc = twenty_closest(reviews, category, embedding)
     
    bow =  bag_words(reviews)
    return_array = []
    for i, review in enumerate(reviews):
        concat_array = tc[i]
        #print(concat_array)
        return_array.append(np.concatenate([bow[i],concat_array]))
    return return_array

def feature_vector_seven(reviews, category, embedding): #Bow Unigrams Embedding - 20 closest, 20 most similar
    reviews_wn = pre_process_wn(reviews)
    tsm = twenty_most_similar(reviews_wn, category)
    
    reviews = pre_process(reviews)
    tc = twenty_closest(reviews, category, embedding)
     
    bow =  bag_words(reviews)
    return_array = []
    for i, review in enumerate(reviews):
        concat_array = tc[i] + tsm[i]
        #print(concat_array)
        return_array.append(np.concatenate([bow[i],concat_array]))
    return return_array
    
 

def get_Y(r):
    return list(r["label"].values())
    

def get_X(r, method = "B", category="food", embedding=None, t = "TFIDF"):
    reviews = r["text"].values()
    if method == "B":
        return bag_words(reviews)
    elif method == 1:
        return feature_vector_one(reviews) 
    elif method == 2:
        return feature_vector_two(reviews)
    elif method == 3:
        return feature_vector_three(reviews)
    elif method == 4:
        return feature_vector_four(reviews, embedding, t)
    elif method == 5:
        return feature_vector_five(reviews, category, t)
    elif method == 6:
        return feature_vector_six(reviews, category, embedding)
    elif method == 7:
        return feature_vector_seven(reviews, category, embedding)
    
    
    
     

def k_fold(X, Y, k = 10, m = "SVC"):
    if m == "SVC":
        model = svm.LinearSVC(dual=True,max_iter = 100000)
    if m == "RF":
        model=RandomForestClassifier(n_estimators=100)
    if m == "KNN":
        model=KNeighborsClassifier(n_neighbors = 20)
    if m =="NB":
        model=GaussianNB()
    if m == "LR":
        model = LogisticRegression()

    X = normalize(X)
    model.fit(X, Y)
    scoring_param = ['accuracy', 'precision', 'recall', 'f1']
    final_scores = {}
    for s in scoring_param:
        scores = cross_val_score(model, X, Y, cv=k, scoring = s)
        final_scores[s] = [np.mean(scores), np.std(scores)] 
    return [model,final_scores]

if __name__ == "__main__":
    print("\n")
    f_dict = pd.read_csv("balanced_food_training_data.csv").to_dict()
    s_dict = pd.read_csv("balanced_service_training_data.csv").to_dict()
    embedding = Word2Vec.load("food_service_embedding.model")

    # Food -- X
    bow_food = get_X(f_dict, "B", "food", embedding)
    one_food = get_X(f_dict, 1, "food", embedding)
    two_food = get_X(f_dict,  2, "food", embedding)
    three_food = get_X(f_dict, 3, "food", embedding, t = "BOW")
    four_food = get_X(f_dict, 4, "food", embedding)
    five_food = get_X(f_dict, 5, "food", embedding, t = "BOW")
    six_food = get_X(f_dict,  6, "food", embedding)
    seven_food = get_X(f_dict,  7, "food", embedding)
    
    food_vectors = [bow_food, one_food, two_food,three_food,four_food,five_food,six_food,seven_food]


    # Food -- Y
    food_labels = get_Y(f_dict)
    
    # Service -- X
    bow_service = get_X(s_dict, "B", "service", embedding)
    one_service = get_X(s_dict, 1, "service", embedding)
    two_service = get_X(s_dict,  2, "service", embedding)
    three_service = get_X(s_dict, 3, "service", embedding, t = "BOW")
    four_service = get_X(s_dict, 4, "service", embedding)
    five_service = get_X(s_dict, 5, "service", embedding, t= "BOW")
    six_service = get_X(s_dict,  6, "service", embedding)
    seven_service = get_X(s_dict,  7, "service", embedding)
    
    service_vectors = [bow_service, one_service, two_service,three_service,four_service,five_service,six_service,seven_service]
    
    # Service -- Y
    
    service_labels = get_Y(s_dict)

        
    # Get Food Results
    
    food_results = {}
    for model in ["SVC"]:
        food_results[model] = [0,0,0,0,0,0,0,0]
        for i,v in enumerate(food_vectors):
            food_results[model][i] = k_fold(v, food_labels, m = model)
            print("Done with " + model + ", food vector " + str(i))
        
    for i,r in enumerate(food_results["SVC"]):
        print(i)
        print(food_results["SVC"][i][1])
    # Get Service Results
    
    service_results = {}
    for model in ["SVC"]:
        service_results[model] = [0,0,0,0,0,0,0,0]

        for i,v in enumerate(service_vectors):
            service_results[model][i] = k_fold(v, service_labels, m = model)
            print("Done with " + model + ", service vector " + str(i))

            
       
    for i,r in enumerate(service_results["SVC"]):
        print(i)
        print(service_results["SVC"][i][1])
    ## Get some misclassified reviews
    
    ## Report results
    
    