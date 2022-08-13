Sentiment Analysis is one of the least impressive and most popular of Machine Learning algorithms. In a nutshell, Sentiment Analysis is a technique wherein we train a program to figure out the *tone* of a message, and further, the sentiment of the public on an issue.

# Algorithm behind Sentiment Analysis

Sentiment Analysis is based on the Naive-Bayes Algorithm, or Naive-Bayes Classifier Algorithm. As students of Conditional Probability will know, the Naive-Bayes Algorithm is used to compute the probability of an event, *given the occurence of another event*. 
To give an example, a person ability to get to work on time is dependent on traffic conditions. Under normal traffic conditions, the probability of a person getting to work on time are high, but they reduce greatly if they're met with a road closure, for instance. Based on a number of given factors, we can predict this probability.

The Naive-Bayes Algorithm is represented as:

![Screenshot 2022-08-13 at 12 05 52 PM](https://user-images.githubusercontent.com/92638241/184471994-fa846c56-cc68-443a-9525-d8d8242f6bee.png)

Here, P(A|B) is called the *Posterior Probability*, P(B|A) is called the *Likelihood*, P(A) is called the *Prior Probability* and P(B) is called the *Marginal Probability*.

While using Naive-Bayes Classifier, our job is to compute the Posterior Probability. To find this, we need to compute Likelihood, Prior and Marginal.


1. **Prior Probability** - Easiest to solve, the Prior Probability is simply the total number of favourable outcomes divided by the total number of possible outcomes. In our example, this is calculated by finding the number of days having abnormal traffic conditions, and dividing that value by the size of the dataset. 

2. **Marginal Probability** - This probability is generally not calculated as it is even across all cases. Thus, this probability is used for normalisation. 

3. **Likelihood** - The main crux of the process, in order to calculate the likelihood probability, we first assume that the day in question is definitely an abnormal traffic day. With this assumption in mind:

    P(B|A) = P(x1|A) . P(x2|A.x1) . P(x3|A.x1.x2) ... P(xn|A.x1.x2...xn-1)

    where x1,x2,x3...xn are *features* of B, in this case, the likelihood of the person getting to work on time. Features may be: alarm clock, redlight  frequency, mood etc. 

    As the above equation indicates, ordinarily, we assume that each of the tasks are somehow interrelated, that is, each factor is dependent on the occurence of the previous ones. This is, however, very computationally expensive, so we adopt the naive approach, and assume all features to be independent of each other. That is, the person's mood is independent of the red lights, which are independent of his alarm clock, etc. Thus, the above equation becomes: 
    
    P(B|A) = P(x1|A) . P(x2|A) . P(x3|A) . P(x4|A).... P(xn|A)

    Computing this, we find the Likelihood. 


# This project

In this project, I have used a dataset of 1.6 Million Tweets (https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download) to train a Naive Bayes ML model to determine the positivity/negativity of a tweet. This dataset gives the positivity/negativity/neutrality of a given tweet on a 0-4 scale, with 0 being negative, 2 neutral and 4 being positive. 
This project involves: 

1. Downloading CSV into a Pandas Dataframe.

2. **Preprocessing** - 
   - Casing - Converting all tweets to lowercase
   - URL/punctuation/hashtags/@ Removal - These characters do not help in the computation and their removal makes tokenization and vectorization easier. I did this using regex and string operations.
   - Stopword removal and Tokenization - Using the Natural Language Toolkit Python library, I removed all the "extra" (or "stopwords") words in the sentence such as "in" "and" "the" etc. which are irrelevant to the computation. This is followed by tokenization, wherein each word is converted into a token, that is, broken up, and made fit for vectorization. The word_tokenize function in the NLTK library was used for this. 
   - Stemming - Using the PorterStemmer() stemmer in the NLTK library, the words are "stemmed". This means that the words with prefixes and suffixes are normalized. Eg. Bigger -> Big, Eating -> Eat etc.
   - Lemmetizing - Using the WordNetLemmatizer() Lemmetizer in the NLTK library, words are lemmetized. In lemmatization we replace words with their ideal morphological equivalent. Eg. Better -> Good, Horrible -> Bad etc.

3. Training and Testing data splitting (test size: 20%) 

4. Vectorization - Used CountVectorizer() method in sklearn for transforming training and test data.

5. Training Data - Used MultinomialNB() method within sklearn.naive_bayes, the first column ("target") is used as the y_train and the sixth column ("tweet") is used as the x_train. 

6. Testing Data - The 20% data reserved for test data is now sent through the now trained model. 
