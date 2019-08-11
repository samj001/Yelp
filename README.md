# 587-Yelp

Our project has three phases.

## Phase 1: Data Management

Collect yelp business.json and review.json. Drop all businesses not in AZ or PA and merge the datasets. Drop all reviews that are not in english and drop all reviews of businesses that are not restaurants. 

merge2.json is the product of data-extract-category.json. 

Please see: https://drive.google.com/file/d/1nrvE_7WbR2VeJqeBTZKas66GDb9ay8hV/view?usp=sharing for the data. 

## Phase 2: Review Classification

Use supervised machine learning techniques to build a review classifier. The classifiers will perform binary classification tasks [Food, Not Food], [Service, Not Service].

We will encode the text data using Bag of Words/TFIDF, Wordnet to classify words such as food specific words, and Word2Vec

We will try the following algorithms: Logistic Regression, SVM on each of food and service.

## Phase 3: More granular studies. 

This section is much more of a work in progress. 

### Phrase 3a: Food Study

We are still working out the details of this analysis. Candidates include
1. Grouping the reviews by restaurant category and then identifying the food items - restaurant pairs that have the best sentiment associated with them
2. Creating a summary that serves as a recommendation for what and where to eat in the area. Ex. In Glendale, try the gnocchi at <restaurant>. It is "delicious, savory, and perfectly cooked." 

### Phase 3b: Service Study

1. Determine what factors influence perceptions of the quality of service. The null hypothesis is that all restaurant categories have similar service. It would be interesting to find that the category of restaurant is influential. 
  a. We can see if service - related reviews have lower star ratings
  b. We can measure sentiment in service related reviews
  
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


