# General purpose imports
import re
import numpy as np 
import pandas as pd 
import matplotlib as mpl
import string

# NLTK imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

# sklearn imports
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# creating stopwords
# nltk.download("punkt")
# nltk.download("wordnet")
stop_words = set(stopwords.words('english'))

def preprocessing(tweet):
    # Casing
    tweet = tweet.lower()

    # URL Removal
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet)

    # Remove punctuation
    tweet = tweet.translate(str.maketrans("","",string.punctuation))

    # Remove hashtag or @
    tweet = re.sub(r"\@\w+|\#","", tweet)

    # Remove stopwords
    tweet_tokens = word_tokenize(tweet)
    filtered_words = []
    for word in tweet_tokens:
        if word not in stop_words:
            filtered_words.append(word)

    # Stemming
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(i) for i in filtered_words]

    # Lemmetizing
    lemmetizer = WordNetLemmatizer()
    lemmetized_words = [lemmetizer.lemmatize(w, pos='a') for w in stemmed_words]

    # Joining
    final = ""
    for word in lemmetized_words:
        final += word
        final += " "

    return final 

df = pd.read_csv("data.csv")
y = df["target"]
x = df["tweet"]
xtr,ytr,xt,yt = train_test_split(
    x, y,
    test_size=0.20,
    random_state = 42
)

xtr_clean = [preprocessing(i) for i in xtr]
xt_clean = [preprocessing(i) for i in xt]

cv = CountVectorizer(ngram_range=(1,2))
xtr_vec = cv.fit_transform(xtr_clean).toarray()
xt_vec = cv.transform(xt_clean).toarray()

mn = MultinomialNB()
mn.fit(xtr_vec,ytr)
