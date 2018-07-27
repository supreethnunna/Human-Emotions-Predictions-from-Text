# Human-Emotions-Predictions-from-Text
PROBLEM STATEMENT:
Construct a Predictive model to predict human emotions based on news paper articles headlines and summaries.  In this dataset we need to classify the document into more than one class, which is a multi output classification. 
OBJECTIVE:
•	Data Set : Sample dataset of emotions and articles
•	Perform Data pre-processing and conduct Data exploration to better understand the characteristics of variables in the Data set. 
•	Perform a detailed analysis and utilize data driven decisions to predict the emotions. 
•	Analyze the influence and relationship of headline and summary in the overall performance.
APPROACH:
Approach taken is to classify a document into set of emotions using supervised techniques and different text representations. Initially performed EDA to understand what type of emotions are usually populated for different type of articles, histogram to see the distribution of number of characters and words in headlines/summary.  Based on initial bi grams of word clouds  worlds like North Korea,  New York, Game of Thrones appear the most in the articles headlines.
Before we build a predictive model we need to transform the text of summary and headline into a feature space. To transform the data into feature space we use two methodologies i.e. TF-IDF and word2vec approach.
TF-IDF:
It is a bag of words approach, gives a product of how frequent this word is in the document multiplied by how unique the word is with respect to entire corpus. Words in the document with a high tfidf score occur frequently in the document and provide the most information about that specific document. The transformed matrix contains different weighted vectors of uni grams, bi grams, tri grams. 
Word2Vec:
Word2vec is an implementation of word embeddings. Used the pre trained word embeddings of Google news vector having a dimension of 300 and selecting adjectives and noun for now in this methodology. Word embedding tries to represent relationships that exist between the individual words by giving them each a vector with same predefined dimension
Once the text is transformed into feature space we use supervised learning approaches Support Vector and Logistic Regression in this case to predict the emotions. Model is being cross validated using cross validation and we use precision, recall and f measure to test the performance of the model.
