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



