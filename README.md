Sentiment Analysis is one of the least impressive and most popular of Machine Learning algorithms. In a nutshell, Sentiment Analysis is a technique wherein we train a program to figure out the *tone* of a message, and further, the sentiment of the public on an issue.

# Algorithm behind Sentiment Analysis

Sentiment Analysis is based on the Naive-Bayes Algorithm, or Naive-Bayes Classifier Algorithm. As students of Conditional Probability will know, the Naive-Bayes Algorithm is used to compute the probability of an event, *given the occurence of another event*. 
To give an example, a person ability to get to work on time is dependent on traffic conditions. Under normal traffic conditions, the probability of a person getting to work on time are high, but they reduce greatly if they're met with a road closure, for instance. Based on a number of given factors, we can predict this probability.

The Naive-Bayes Algorithm is represented as:

                P(A|B) = P(B|A) . P(A)
                            P(B)

Here, P(A|B) is called the *Posterior Probability*, P(B|A) is called the *Likelihood*, P(A) is called the *Prior Probability* and P(B) is called the *Marginal Probability*.

While using Naive-Bayes Classifier, our job is to compute the Posterior Probability. To find this, we need to compute Likelihood, Prior and Marginal. 

1. Prior Probability - Easiest to solve, 