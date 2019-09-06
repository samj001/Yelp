# Logistic Model Using Semantic Features to Analyze Yelp Reviews

## INTRODUCTION

Yelp users are not a monolith. Some are looking to impress a date and want to ensure they have a pleasant experience. Others are adventurous foodies happy to be completely ignored by the waitstaff as long as the food tastes delicious. However, Yelp does not tailor the reviews it displays to its users. Instead, people with predictable preferences are left to sift through reviews until they find enough that are relevant to them. This is the motivation for our work. We set out to find a way to sort reviews in to food and service categories. To accomplish this goal, we extracted a set of semantic features for each review that we hypothesized would aid in classification. In the best performing case, we built a word embedding trained on 50,000 reviews and weighted words by their tokens’ cosine-similarities to words emblematic of food and service. We then trained binary classifiers – one for food and one for service. Our best classifiers approached human labeling performance.  
  
Our research questions are as follows: Can we build accurate food and service classifiers for Yelp reviews? And does the inclusion of semantic features improve classification when compared with baseline text representations such as TFIDF and bag-of-words?   
  
## DATA  
  
We used the Yelp review dataset to test our hypotheses. The datasets contain information about millions of businesses and six million reviews of those businesses. We first merged the business and review datasets, and our analysis focused on restaurants, we removed reviews of businesses that were not restaurants. Additionally, to make data processing more manageable, we limited the dataset by dropping rows not associated with restaurants in Pennsylvania, Nevada, or Arizona, the states with the highest volume of reviews in the dataset. We further limited the dataset to the first 150,000 reviews.  
  

## Labeling and Preprocessing   

To train our model, we labeled 1,200 reviews according to whether they made comments about a restaurant’s food, service, both, or none. On the 100 reviews multiple labelers examined, we achieved the following inter-labler agreement rates: For whether a review made a comment about food or not, we agreed 97% of the time. For whether a review made a comment about service or not, we agreed 87% of the time.   
  
We made a number of decisions in preprocessing reviews. First, we dropped stop-words from the Natural Language Processing Toolkit’s publicly available list of stop words. Then we stripped the review of all punctuation and stemmed each token in each review. While these pre-processing steps arguably strip the review data of important context, our results demonstrate that the token-level information was sufficient to make accurate predictions about reviews’ classes.   
  

## Phrase 3a: Representing Reviews  
  
For our three baselines, we represented reviews as a bag-of-words,  TFIDF-weighted unigrams, and TFIDF-weighted unigrams and bigrams. We then created six additional feature vectors described in detail below. Before explaining our contribution, we describe one metric: Wu-Palmer Similarity.  
### Wu-Palmer

Recall that WordNet represents the English language as a tree structure. Intuitively, the distance between words in the tree would provide information about their semantic similarities to one another. Wu and Palmer formalized this notion in 1994. 
  
<p align="center">SimWP(u,v)=2D/(Lu+Lv) </p>  
  
Where u and v represent nodes in WordNet, and D is the distance from the root to the lowest common ancestor of u and v, which is analogous to their mutual hypernym. Lu and Lv represent the distances from u and v to their lowest common parent ancestor respectively.  

### Review Level Features  
  
Our fourth and fifth feature vectors incorporated reviews’ aggregate similarities to word emblematic of food and service.   
  
For feature vector four, for each review, we concatenated the review’s TFIDF-weighted unigrams to a two-dimensional vector W. W1 contained the average of the review’s noun’s Wu-Palmer similarities to the word food. W2 was more complicated. For each noun in the review, we calculated SimWP(noun,”service”) and SimWP(noun,”waiter”). We took the minimum of those two values and appended it to a vector. Then W2 was the mean of that vector.     
  
For feature-vector five, we used a word-embedding trained on 50,000 reviews. For each review, we also concatenated the review’s TFIDF-weighted unigrams to a two-dimensional vector W. In this case, W1 contained the average of the review’s tokens cosine similarities (in the embedding) to words emblematic of food and W2  contained the average of the review’s tokens cosine similarities to words emblematic of service. The words emblematic of food and service were initially selected by hand and expanded by using the embedding.  
  
  
### Token Level Features

Our sixth thru ninth feature vectors used tokens’ similarities to words emblematic of food and service. For the sixth vector, when using our food classifier, for each review, we generated a twenty-dimensional vector representing the Wu Palmer similarities of the twenty words closest to “food.” When using our service classifier, we generated the same vector using the words “service” and “waiter” as above. We for each review, we concatenated its “top-20” vector to the review’s bag of words. Our seveth vector was built with an analogous approach, but it used the embedding to calculate similarity instead of WordNet. For each of these, we handled reviews that did not have enough words by padding the top-20 vector with its mean. The eighth feature vector was the review’s bag of words with both top-20s.   

Note that there is a significant difference between what is being measured in this section and the previous one. When we compute the review level features, we are returning similarities to both food and service for each review. In this section, we the vector only provides information about one of the two depending on the classification task. If the aggregate feature performs better, we can argue that knowing a review hardly covers food means it is more likely to be about service and vice-versa.  

Our ninth feature vector was more complicated. We used the embedding to calculate the cosine similarity of each word to food words or service words depending on the classification task. We then normalized the distances as follows:  
  
cosine_sim_normalized(token, class_words) = (cosine_sim(token, class_words) - min) / (max - min)  
  	  
The feature vector is then a cosine_sim_normalized weighted bag of words.   
  
## RESULTS AND DISCUSSION 
  
For feature vectors with asterisks, logistic regression outperformed the rest of the models. Otherwise, the linear SVM was the best model.  

### Food Classifier

![image](https://github.com/samj001/Yelp/blob/master/image/food%20classifier.png)
 
 


### Service Classifier
![image](https://github.com/samj001/Yelp/blob/master/image/service%20classifier.png)
  
The cosine-similarity weighted bag of words was the overall best-performing model for both tasks. Semantic features improved classification accuracy. 

## Misclassification Examples
  

There were generally two categories of misclassified reviews. First, there were reviews that were mislabeled by the labelers that our classifiers labeled correctly. We see the humor in this, but recognize that labeling carelessness is a weakness of our study that we will work to address. Perhaps the labeling interface could be improved. Our classifiers also struggled with short reviews. For example, it predicted that the review “Great food and great atmosphere for a family. Very good value!” was about both food and service.  

# CONCLUSION

We conclude that semantic features drawn from WordNet and word embeddings trained on reviews can, in fact, aid in review classification. There are many use cases for our research. For example, Yelp could filter reviews according to category and tailor star ratings to users’ interests. Additionally, restaurants could more systematically determine the areas of improvement if they were able to filter reviews. 


