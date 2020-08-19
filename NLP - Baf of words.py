# -*- coding: utf-8 -*-
"""
Created on Wed May 13 10:22:17 2020

@author: Shrikant Agrawal
"""

import nltk

paragraph = """I have three visions for India. In 3000 years of our history, people from all over 
               the world have come and invaded us, captured our lands, conquered our minds. 
               From Alexander onwards, the Greeks, the Turks, the Moguls, the Portuguese, the British,
               the French, the Dutch, all of them came and looted us, took over what was ours. 
               Yet we have not done this to any other nation. We have not conquered anyone. 
               We have not grabbed their land, their culture, 
               their history and tried to enforce our way of life on them. 
               Why? Because we respect the freedom of others.That is why my 
               first vision is that of freedom. I believe that India got its first vision of 
               this in 1857, when we started the War of Independence. It is this freedom that
               we must protect and nurture and build on. If we are not free, no one will respect us.
               My second vision for India’s development. For fifty years we have been a developing nation.
               It is time we see ourselves as a developed nation. We are among the top 5 nations of the world
               in terms of GDP. We have a 10 percent growth rate in most areas. Our poverty levels are falling.
               Our achievements are being globally recognised today. Yet we lack the self-confidence to
               see ourselves as a developed nation, self-reliant and self-assured. Isn’t this incorrect?
               I have a third vision. India must stand up to the world. Because I believe that unless India 
               stands up to the world, no one will respect us. Only strength respects strength. We must be 
               strong not only as a military power but also as an economic power. Both must go hand-in-hand. 
               My good fortune was to have worked with three great minds. Dr. Vikram Sarabhai of the Dept. of 
               space, Professor Satish Dhawan, who succeeded him and Dr. Brahm Prakash, father of nuclear material.
               I was lucky to have worked with all three of them closely and consider this the great opportunity of my life. 
               I see four milestones in my career"""


# Cleaning the text - removing stop words, removing , ., lamitizationm, stemming, lower the sentenses
import re         # regular expression library
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer   # library for stemmer
from nltk.stem import WordNetLemmatizer      # library for Lemmatizer# library for stemmer

# object creation
ps = PorterStemmer()
wordnet = WordNetLemmatizer()
sentenses = nltk.sent_tokenize(paragraph)   # Create sentenses

corpus = []   # created new list, after cleaning the para we will store details into it
for i in range(len(sentenses)):    #Iterating through each and every sentenses. It will run 31 times
    review = re.sub('[^a-zA-Z]', ' ', sentenses[i])  # Removing .,!? and other charactes apart from a-z and A - Z and replacing it with blanck spaces
    review = review.lower()        # lowering each and every sentenses
    review = review.split()        # list of words
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# you can see all special characters and stop keywords are removed from the sentences
# when you compare corpus with sentenses you will see history gets converted to histori, people to popl which does not makes any sence because we used stmmer

# Now lets use lemmatizer and check the output which is meaningful info in all lower case
corpus1 = []   # created new list, after cleaning the para we will store details into it
for j in range(len(sentenses)):    #Iterating through each and every sentenses. It will run 31 times
    review1 = re.sub('[^a-zA-Z]', ' ', sentenses[j])  # Removing .,!? and other charactes apart from a-z and A - Z and replacing it with blanck spaces
    review1 = review1.lower()        # lowering each and every sentenses
    review1 = review1.split()        # list of words
    review1 = [wordnet.lemmatize(word) for word in review1 if not word in set(stopwords.words('english'))]
    review1 = ' '.join(review1)
    corpus1.append(review1)


# Creating the Bag of words model
from sklearn.feature_extraction.text import CountVectorizer  # responsible for creating BOW document matrix
cv = CountVectorizer()
x = cv.fit_transform(corpus1).toarray()   # Converting it to array so we can see it properly

# x output is now combination of 0,1,2.. numbers we can apply ML algorithm on it. It has 31 sentenses and 114 variables