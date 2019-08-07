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

