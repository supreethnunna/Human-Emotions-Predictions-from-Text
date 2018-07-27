
# coding: utf-8

# In[7]:

import json
import pandas as pd


# ### <font color=Black> IMPORT VARIOUS PACAKGES</font>

# In[8]:

import matplotlib
import numpy as np
import matplotlib.pyplot as plt


# In[9]:

from wordcloud import WordCloud


# In[4]:

import sys
import os
import pickle


# In[10]:

import sklearn
import gensim


# In[6]:

from argparse import ArgumentParser
from gensim.models import KeyedVectors
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report

from nltk import sent_tokenize
from nltk import pos_tag
from nltk import map_tag
from nltk import word_tokenize
from nltk.corpus import stopwords


# In[11]:

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# In[12]:

#Create a empty list or file
df = []


# ### <font color=Black> Load the JSON File and create a Data Frame </font>

# In[13]:

#Loop through every record and append to the file
for line in open('C:/Users/Supreeth/Downloads/homework.json', 'r'):
    df.append(json.loads(line))


# In[9]:

df


# In[14]:

#Covnert to a data frame 
df1 = pd.DataFrame(df)


# ### <font color=Black> Perform Data Cleaning and required Data Transformation</font>

# In[15]:

df1.shape


# In[16]:

df1.dtypes


# In[17]:

df1


# In[18]:

df1['headline'].head(10)


# In[19]:

##Removing Trailing and Leading White Spaces from the String
df1['summary'] = df1['summary'].str.strip()


# In[20]:

df1.describe()


# In[21]:

## Find Number of missing values
df1.isnull().sum()


# In[22]:

## Create a column list to exclude records which have null or missing values
col_list = []
string = 'emotion_'
for i in range(0,10):
        col_list.append(string + str(i))
print(col_list)


# In[23]:

#Drop recrods are missing
df2 = df1.dropna(subset=col_list)


# In[24]:

#We are excluding the missing records where emotion values are zero . Since there are less than 
#2% of missing records we are using any imputation techniques like mean, median,mode or KNN or clustering and other techinques
df2.shape


# In[25]:

df2.describe()


# In[26]:


##Cleaning all the bad Data
df3 = df2[(df2['emotion_0'] != 'cat') & (df2['emotion_0'] != 'fnord') & (df2['emotion_0'] != '-2') & (df2['emotion_0'] != '-1')&
   (df2['emotion_1'] != 'cat') & (df2['emotion_1'] != 'fnord') & (df2['emotion_1'] != '-2') & (df2['emotion_1'] != '-1')&
    (df2['emotion_2'] != 'cat') & (df2['emotion_2'] != 'fnord')& (df2['emotion_2'] != '-2') & (df2['emotion_2'] != '-1')&
    (df2['emotion_3'] != 'cat') & (df2['emotion_3'] != 'fnord')& (df2['emotion_3'] != '-2') & (df2['emotion_3'] != '-1')&
    (df2['emotion_4'] != 'cat') & (df2['emotion_4'] != 'fnord')& (df2['emotion_4'] != '-2') & (df2['emotion_4'] != '-1')&
    (df2['emotion_5'] != 'cat') & (df2['emotion_5'] != 'fnord')& (df2['emotion_5'] != '-2') & (df2['emotion_5'] != '-1')&
    (df2['emotion_6'] != 'cat') & (df2['emotion_6'] != 'fnord')& (df2['emotion_6'] != '-2') & (df2['emotion_6'] != '-1')&
    (df2['emotion_7'] != 'cat') & (df2['emotion_7'] != 'fnord')& (df2['emotion_7'] != '-2') & (df2['emotion_7'] != '-1')&
    (df2['emotion_8'] != 'cat') & (df2['emotion_8'] != 'fnord')& (df2['emotion_8'] != '-2') & (df2['emotion_8'] != '-1')&
    (df2['emotion_9'] != 'cat') & (df2['emotion_9'] != 'fnord') & (df2['emotion_9'] != '-2') & (df2['emotion_9'] != '-1')&
    (df2['headline'] != '-2') & (df2['headline'] != '-1')   &
    (df2['summary'] != '-2') & (df2['summary'] != '-1')
         
         ]


# In[27]:

df3.describe()


# In[28]:

df3.shape


# In[29]:

df3.info()


# In[30]:

df4 = df3


# In[31]:

df4[col_list] = df4[col_list].apply(pd.to_numeric)


# In[35]:

df4.to_csv('C:/Users/Supreeth/Downloads/ds.csv', sep=',')


# ### <font color=Black> Explatory Data Analysis </font>

# In[32]:

df_emotions = df4.drop(['headline', 'summary','worker_id'], axis=1)
counts = []
categories = list(df_emotions.columns.values)
for i in categories:
    counts.append((i, df_emotions[i].sum()))
df_stats = pd.DataFrame(counts, columns=['emotions', '#articles'])
df_stats


# In[33]:

#Emotion 9 is present for every article
df_stats.plot(x='emotions', y='#articles', kind='bar', legend=False, grid=True, figsize=(15, 8))
plt.show()


# In[37]:

#Text Analysis
headline_text = pd.Series(df4['headline'].tolist()).astype(str)


# In[48]:

headline = headline_text.apply(len)


# In[50]:

#Most of the characters in the headlines are between 50 - 75 characters
plt.hist(headline, bins=120, range=[0, 120])
plt.title('Histogram of Character Count in Headlines')
plt.xlabel('Character Count')
plt.ylabel('Probability')
plt.show()
print('mean {:.2f} std {:.2f} max {:.2f} '.format(headline.mean(),headline.std(), headline.max()))


# In[51]:

summary_text = pd.Series(df4['summary'].tolist()).astype(str)
summary = summary_text.apply(len)
plt.hist(summary, bins=250, range=[0, 250])
plt.title('Histogram of Character Count of Summary')
plt.xlabel('Character Count')
plt.ylabel('Probability')
plt.show()
print('mean {:.2f} std {:.2f} max {:.2f} '.format(summary.mean(),summary.std(), summary.max()))


# In[52]:

headline = headline_text.apply(lambda x: len(x.split(' ')))

plt.hist(headline, bins=25, range=[0, 25])
plt.title('Hist of No of Words')
plt.xlabel('Number of words')
plt.ylabel('Probability')
plt.show()


print('mean {:.2f} std {:.2f} max {:.2f}'.format(headline.mean(), headline.std(), headline.max()))


# In[55]:

summary = summary_text.apply(lambda x: len(x.split(' ')))
plt.hist(summary, bins=50, range=[0, 50])
plt.title('Histogram of Word Count of Summary')
plt.xlabel('Word Count')
plt.ylabel('Probability')
plt.show()
print('mean {:.2f} std {:.2f} max {:.2f} '.format(summary.mean(),summary.std(), summary.max()))


# In[53]:

from wordcloud import WordCloud
cloud = WordCloud(width=1440, height=1080).generate(" ".join(headline_text.astype(str)))
plt.figure(figsize=(20,20))
plt.imshow(cloud)
plt.axis('off')
plt.show()


# In[54]:

qmarks = np.mean(headline_text.apply(lambda x: '?' in x))
got = np.mean(headline_text.apply(lambda x: 'Game of Thrones' in x))
ny = np.mean(headline_text.apply(lambda x: 'New York' in x))
nk = np.mean(headline_text.apply(lambda x: 'North Korea' in x))
homes = np.mean(headline_text.apply(lambda x: 'Homes' in x))

print('Headlines with question marks: {:.2f}%'.format(qmarks * 100))
print('Headlines with Game of Thornes: {:.2f}%'.format(got * 100))
print('Headlines with New York: {:.2f}%'.format(ny * 100))
print('Headlines with North Korea: {:.2f}%'.format(nk * 100))
print('Headlines with Home: {:.2f}%'.format(homes * 100))


# ### <font color=Black> Building necessary functions for Word2Vec and TF-IDF Approach </font>

# In[99]:

def tag_pos(x):
    sentences = sent_tokenize(x)
    sents = []
    for s in sentences:
        text = word_tokenize(s)
        pos_tagged = pos_tag(text)
        simplified_tags = [
            (word, map_tag('en-ptb', 'universal', tag)) for word, tag in pos_tagged]
        sents.append(simplified_tags)
    return sents


# In[58]:

def post_tag_documents(df4):
    x_data = []
    y_data = []
    total = len(df4['headline'].as_matrix().tolist())
    headline = df4['headline'].as_matrix().tolist()
    emotions = df4.drop(['headline', 'summary', 'worker_id'], axis=1).as_matrix()
    for i in range(len(headline)):
        sents = tag_pos(headline[i])
        x_data.append(sents)
        y_data.append(emotions[i])
        i += 1
        if i % 5000 == 0:
            print(i, "/", total)

    return x_data, y_data


# In[79]:

stop_words = set(stopwords.words('english'))


# In[105]:

def word2vec(x_data, pos_filter):

 
    google_vecs = KeyedVectors.load_word2vec_format(
        'C:/Users/Supreeth/Desktop/GoogleNews-vectors-negative300.bin', binary=True, limit=200000)

    x_data_embeddings = []
    total = len(x_data)
    processed = 0
    for tagged_plot in x_data:
        count = 0
        doc_vector = np.zeros(300)
        for sentence in tagged_plot:
            for tagged_word in sentence:
                if tagged_word[1] in pos_filter:
                    try:
                        doc_vector += google_vecs[tagged_word[0]]
                        count += 1
                    except KeyError:
                        continue

        doc_vector /= count
        if np.isnan(np.min(doc_vector)):
            continue

        x_data_embeddings.append(doc_vector)

        processed += 1
        if processed % 10000 == 0:
            print(processed, "/", total)

    return np.array(x_data_embeddings)


# ### <font color=Black>Building Target and Predictors Datasets</font>

# In[80]:

#Removing the prediction of emotion_9 as it is present for all records
data_x = df4[['headline']].as_matrix()
data_y = df4.drop(['headline', 'summary', 'worker_id','emotion_9'], axis=1).as_matrix()


# In[75]:

data_y


# ### <font color=Black>Predictive Model using TF-IDF </font>

# In[82]:

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_x,data_y, test_size=0.33,random_state=42)


# In[83]:

train_x = [x[0].strip() for x in x_train.tolist()]
test_x = [x[0].strip() for x in x_test.tolist()]


# ### <font color=Black>TF-IDF using Logistic Regression </font>

# In[84]:

parameters = {
    'tfidf__max_df': (0.25, 0.5, 0.75),
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    "clf__estimator__C": [0.01, 0.1, 1],
    "clf__estimator__class_weight": ['balanced', None],
}


# ### <font color=Black>Incorporating Pipe Line form sklearn to enable different transformation together </font>

# In[85]:

pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
            ])


# In[86]:

grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=3, verbose=10)
grid_search_tune.fit(train_x, y_train)


# ### <font color=Black>Precision, Recall , F-measure of the Model </font>

# In[87]:

best_clf = grid_search_tune.best_estimator_
predictions = best_clf.predict(test_x)


# In[93]:

emotions = df4.drop(['headline', 'summary', 'worker_id'], axis=1).as_matrix()


# In[95]:

print(classification_report(y_test, predictions))


# ### <font color=Black>Using Summary instead of Headline </font>

# In[158]:

#Removing the prediction of emotion_9 as it is present for all records
data_x = df4[['summary']].as_matrix()
data_y = df4.drop(['headline', 'summary', 'worker_id','emotion_9'], axis=1).as_matrix()


# In[170]:

data_x


# In[159]:

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_x,data_y, test_size=0.33,random_state=42)


# In[160]:

train_x = [x[0].strip() for x in x_train.tolist()]
test_x = [x[0].strip() for x in x_test.tolist()]


# In[161]:

parameters = {
    'tfidf__max_df': (0.25, 0.5, 0.75),
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    "clf__estimator__C": [0.01, 0.1, 1],
    "clf__estimator__class_weight": ['balanced', None],
}


# In[162]:

pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
            ])


# In[163]:

grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=3, verbose=10)
grid_search_tune.fit(train_x, y_train)


# In[164]:

best_clf = grid_search_tune.best_estimator_
predictions = best_clf.predict(test_x)


# In[165]:

print(classification_report(y_test, predictions))


# ### <font color=Black>TF-IDF using SVM </font>

# In[154]:

#Computationally Implementing SVM is very expensive. O(n) of SVM is high. 
#Ideally would implement SVM on the server due to avialbility of High RAM and processing power on the servers
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stop_words)),
    ('clf', OneVsRestClassifier(LinearSVC()),
)])
parameters = {
    'tfidf__max_df': (0.25, 0.5, 0.75),
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    "clf__estimator__C": [0.01, 0.1, 1],
    "clf__estimator__class_weight": ['balanced', None],
}


# ### <font color=Black>Word2Vec Approach </font>

# In[100]:

x_data, y_data = post_tag_documents(df4)


# In[102]:

pos_filter = ['NOUN', 'ADJ']


# In[106]:

x_embeddings = word2vec(x_data, pos_filter)


# In[125]:

x_embeddings.shape


# In[128]:

x_embeddings


# In[129]:

y_data


# In[130]:

y_data_new = y_data[:85339,:]


# In[131]:

y_data_new.shape


# In[126]:

y_data.shape


# In[107]:

y_data = np.array(y_data)


# In[117]:

x_data


# In[115]:

y_data


# In[132]:

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_embeddings,y_data_new, test_size=0.33,random_state=42)


# In[133]:

pipeline = Pipeline([
                
                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
            ])


# In[134]:

parameters = {
    "clf__estimator__C": [0.01, 0.1, 1],
    "clf__estimator__class_weight": ['balanced', None],
}


# In[135]:

grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=3, verbose=10)
grid_search_tune.fit(x_train, y_train)


# ### <font color=Black>Word Embeddings Method shows very poor results compared to TF-IDF. Exploring Recurrent Neural Nets with Word embeddings might be a better option </font>

# In[136]:

best_clf = grid_search_tune.best_estimator_
predictions = best_clf.predict(x_test)


# In[113]:

pipeline = Pipeline([
        ('clf', OneVsRestClassifier(SVC(), n_jobs=1)),
    ])


# In[112]:

parameters = [

        {'clf__estimator__kernel': ['rbf'],
         'clf__estimator__gamma': [1e-3, 1e-4],
         'clf__estimator__C': [1, 10]
        },

        {'clf__estimator__kernel': ['poly'],
         'clf__estimator__C': [1, 10]
        }
         ]


# ### <font color=Black>Best Coding Practise is to create various functions for the models </font>

# In[180]:

def model_svm(x_data, y_data):
    x_train, x_test, y_train, y_test = train_test_split(data_x,data_y, test_size=0.33,random_state=42)
    train_x = [x[0].strip() for x in x_train.tolist()]
    test_x = [x[0].strip() for x in x_test.tolist()]
    pipeline = Pipeline([
        ('clf', OneVsRestClassifier(SVC(), n_jobs=1)),
    ])
    parameters = [

        {'clf__estimator__kernel': ['rbf'],
         'clf__estimator__gamma': [1e-3, 1e-4],
         'clf__estimator__C': [1, 10]
        },

        {'clf__estimator__kernel': ['poly'],
         'clf__estimator__C': [1, 10]
        }
         ]
    model_output(x_train, y_train, x_test, y_test, parameters, pipeline)


# In[181]:

def model_lr(x_data, y_data):
    x_train, x_test, y_train, y_test = train_test_split(data_x,data_y, test_size=0.33,random_state=42)
    train_x = [x[0].strip() for x in x_train.tolist()]
    test_x = [x[0].strip() for x in x_test.tolist()]
    pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
            ])
    parameters = {
    'tfidf__max_df': (0.25, 0.5, 0.75),
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    "clf__estimator__C": [0.01, 0.1, 1],
    "clf__estimator__class_weight": ['balanced', None],
    }
    
    model_output(x_train, y_train, x_test, y_test, parameters, pipeline)


# In[179]:

def model_output(train_x, train_y, test_x, test_y, parameters, pipeline):
    grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=3, verbose=10)
    grid_search_tune.fit(train_x, train_y)
    best_clf = grid_search_tune.best_estimator_
    predictions = best_clf.predict(test_x)
    print(classification_report(y_test, predictions))


# In[182]:

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_x,data_y, test_size=0.33,random_state=42)


# In[184]:

train_x = [x[0].strip() for x in x_train.tolist()]
test_x = [x[0].strip() for x in x_test.tolist()]


# In[185]:

parameters = {
    "clf__estimator__C": [0.01, 0.1, 1],
    "clf__estimator__class_weight": ['balanced', None],
}


# In[187]:

pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
            ])


# In[188]:

grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=3, verbose=10)
grid_search_tune.fit(train_x, y_train)


# In[189]:

best_clf = grid_search_tune.best_estimator_
predictions = best_clf.predict(test_x)


# In[190]:

print(classification_report(y_test, predictions))


# In[62]:

#import nltk


# In[63]:

#nltk.download()

