# -*- coding: utf-8 -*-
"""
Created on Wed May 13 09:27:37 2020

@author: Shrikant Agrawal
"""

#importting the dataset

import pandas as pd

messages = pd.read_csv('SMSSpamCollection',sep= '\t', names = ['label', 'message'])  # tab seperated file checked in notepad, adding two column names


# Data cleaning and preprocessing
import re
import nltk
#nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(corpus).toarray()

# In the output we have 6296 columns, so here we have to take most frequest elements ie words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)   # top 5000 columns will get selected
x = cv.fit_transform(corpus).toarray()

y=pd.get_dummies(messages['label'])   # label is our indepndent variable. Converting it to numbers because system won't understand text
y=y.iloc[:,1].values                 # picking up only second column, because when spam is 0 means ham

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state=0)

# Training model using Naive Byes Classifier - it works well with NLP data
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(x_train, y_train)

y_pred = spam_detect_model.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)












