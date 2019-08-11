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
  
RESULTS AND DISCUSSION 

For feature vectors with asterisks, logistic regression outperformed the rest of the models. Otherwise, the linear SVM was the best model.

Food Classifier

 
 
FV 0
FV 1*
FV 2*
FV 3
FV 4*
FV 5*
FV 6
FV 7
FV 8*
Accuracy 
0.791
(+/- 0.06)
0.82
(+/- 0.12)
0.809
(+/- 0.07)
0.838
(+/- 0.10)
0.87
(+/- 0.08)
0.83
(+/- 0.07)
0.812
(+/- 0.07)
0.820
(+/- 0.09)
0.86
(+/- 0.1)
Precision
0.818
(+/- 0.08)
0.818
(+/- 0.10)
0.78
(+/- 0.09)
0.851
(+/- 0.13)
0.88
(+/- 0.10)
0.85
(+/- 0.08)
0.824
(+/- 0.11)
0.830
(+/- 0.11)
0.89
(+/- 0.11)
Recall
0.769
(+/- 0.14)
0.827
(+/- 0.08)
0.89
(+/- 0.09)
0.842
(+/- 0.10)
0.87
(+/- 0.08)
0.81
(+/- 0.12)
0.813
(+/- 0.11)
0.820
(+/- 0.12)
0.85
(+/- 0.11))
F1-score
0.781
(+/- 0.08)
0.816
(+/- 0.05)
0.81
(+/- 0.06)
0.841
(+/- 0.09)
0.87
(+/- 0.07)
0.82
(+/- 0.09)
0.812
(+/- 0.09)
0.820
(+/- 0.09)
0.86
(+/- 0.08)


Service Classifier

 
 
FV 0
FV 1
FV 2
FV 3
FV 4
FV 5
FV 6
FV 7
FV 8
Accuracy 
0.801
(+/- 0.05)
0.780
(+/- 0.05)
0.796
(+/- 0.05)
0.787
(+/- 0.06)
0.795
(+/- 0.07)
0.821
(+/- 0.05)
0.831
(+/- 0.05)
0.833
(+/- 0.05)
0.845
(+/- 0.03)
Precision
0.826
(+/- 0.05)
0.801
(+/- 0.06)
0.811
(+/- 0.07)
0.808
(+/- 0.07)
0.825
(+/- 0.07)
0.850
(+/- 0.06)
0.841
(+/- 0.05)
0.843
(+/- 0.06)
0.870
(+/- 0.05)
Recall
0.765
(+/- 0.10)
0.748
(+/- 0.09)
0.783
(+/- 0.09)
0.755
(+/- 0.08)
0.753
(+/- 0.09)
0.786
(+/- 0.08)
0.819
(+/- 0.09)
0.821
(+/- 0.09)
0.813
(+/- 0.08)
F1-score
0.791
(+/- 0.07)
0.770
(+/- 0.06)
0.792
(+/- 0.05)
0.779
(+/- 0.06)
0.784
(+/- 0.07)
0.813
(+/- 0.06)
0.827
(+/- 0.06)
0.830
(+/- 0.06)
0.837
(+/- 0.04)

The cosine-similarity weighted bag of words was the overall best-performing model for both tasks. Semantic features improved classification accuracy. 

Misclassification Examples

There were generally two categories of misclassified reviews. First, there were reviews that were mislabeled by the labelers that our classifiers labeled correctly. We see the humor in this, but recognize that labeling carelessness is a weakness of our study that we will work to address. Perhaps the labeling interface could be improved. Our classifiers also struggled with short reviews. For example, it predicted that the review “Great food and great atmosphere for a family. Very good value!” was about both food and service.  

7 CONCLUSION

We conclude that semantic features drawn from WordNet and word embeddings trained on reviews can, in fact, aid in review classification. There are many use cases for our research. For example, Yelp could filter reviews according to category and tailor star ratings to users’ interests. Additionally, restaurants could more systematically determine the areas of improvement if they were able to filter reviews. 


