# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 

import nltk
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier

from wordcloud import wordcloud,STOPWORDS

from subprocess import check_output

import mlflow 


class Sentiment:
   def __init__(self):
    
       data = pd.read_csv(r'C:\Users\user\Downloads\Sentiment.csv')

       mlflow.log_param('data',data)

        # get the tweet text and related sentiments 

       data = data[['text','sentiment']]

        # Splitting the dataset into train and test set

       self.train, self.test = train_test_split(data,test_size = 0.1)

       self.train = train[train.sentiment != "Neutral"]

       self.train_pos = train[ train['sentiment'] == 'Positive']
       self.train_pos = train_pos['text']
       self.train_neg = train[ train['sentiment'] == 'Negative']
       self.train_neg = train_neg['text']

       self.tweets = []
       self.stopwords_set = set(stopwords.words("english"))

       for index, row in train.iterrows():
           self.words_filtered = [e.lower() for e in row.text.split() if len(e) >= 3]
           self.words_cleaned = [word for word in self.words_filtered
           if 'http' not in word
              and not word.startswith('@')
              and not word.startswith('#')
              and word != 'RT']
           self.words_without_stopwords = [word for word in self.words_cleaned if not word in self.stopwords_set]
           self.tweets.append((self.words_without_stopwords, row.sentiment))


       self.test_pos = test[ test['sentiment'] == 'Positive']
       self.test_pos = self.test_pos['text']
       self.test_neg = test[ test['sentiment'] == 'Negative']
       self.test_neg = self.test_neg['text']


   def run(self):
       self.train_model()
       self.test_model()

# Extracting word features


def get_words_in_tweets(tweets):
    self.all = []
    for (words, sentiment) in self.tweets:
        self.all.extend(words)
    return all

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    features = wordlist.keys()
    return features



def extract_features(document):
    document_words = set(document)
    features = {}
    
    w_features = get_word_features(get_words_in_tweets(self.tweets))
    
    for word in w_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


# Training the Naive Bayes classifier with NLTK.NaiveBayesClassifier
 

def train_model():    
       
    training_set = nltk.classify.apply_features(extract_features,tweets)

    classifier = nltk.NaiveBayesClassifier.train(training_set)



# Testing the Accuracy of the Classifier

def test_model():

    neg_cnt = 0 

    pos_cnt = 0

    for obj in test_neg: 
        res =  classifier.classify(extract_features(obj.split()))
    if(res == 'Negative'):
       neg_cnt = neg_cnt + 1
          
    for obj in test_pos: 
        res =  classifier.classify(extract_features(obj.split()))
    if(res == 'Positive'):
       pos_cnt = pos_cnt + 1

    mlflow.log_metric('pos_cnt',pos_cnt)
    
    mlflow.log_metric('neg_cnt',neg_cnt) 

    print('[Negative]: %s/%s '  % (len(test_neg),neg_cnt))        

    print('[Positive]: %s/%s '  % (len(test_pos),pos_cnt))    


def main():
    
    model = Sentiment()
    model.run()


if __name__ == "__main__":
  main()
  