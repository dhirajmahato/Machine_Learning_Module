Here, I will add Machine Learning Models and usage and Explanations

## Types/family of ML
based on Human supervision:
1. Supervised
2. Unsupervised
3. Semisupervised
4. Reinforcement

based on techniques
1. Ensemble Learning Algorithms: Bagging, Boosting, stacking
2. Deep Learning Algorithms: CNN, RNN
3. Bayesian Learning Algorithms : Naive Bayes, Bayesian Networks
4. Instance-Based Learning Algorithms: k-Nearest Neighbors (k-NN), Locally Weighted Regression (LWR)
5. Clustering Algorithms : K-Means, Hierarchical clustering
6. Dimensionality Reduction Algorithms: PCA (linear), t-SNE(non linear)

based on data ingestion
1. batch processing
2. online processing

based on content
1. **Discriminative**: A Discriminative model ‌models the decision boundary between the classes (conditional probability distribution p(y|x)).
    - ‌Logistic regression, SVMs, ‌CNNs, RNNs, Nearest neighbours.
2. **Generative**: A Generative Model ‌explicitly models the actual distribution of each class (joint probability distribution p(x,y)).
    - Use Bayes rule to calculate P(Y |X)
    - Naïve Bayes, Bayesian networks, Markov random fields, AutoEncoders, GANs.

based on parameter
1. **Parametric**: parametric model summarizes data with a set of fixed-size parameters (independent on the number of instances of training). Eg: Linear, Logistic Regression, linear SVM (wTx + b = 0), Linear Discriminant Analysis, Perceptron, Naive Bayes, Simple Neural Networks.
2. **Non-parametric**: which do not make specific assumptions about the type of the mapping function. The word nonparametric does not mean that the value lacks parameters existing in it, but rather that the parameters are adjustable and can change. eg:  k-Nearest Neighbors, Decision Trees, SVMs.

A **paramter** is something that is estimated from the training data and change (learnt) while training a model. They can be weights, coefficients, support vectors etc.

**Maximumm Liklihood Estimation (MLE)** is a method that determines values of the parameters of a model such that they maximise the likelihood of observed data given a probability distribution.

## ML Table
| Machine Learning Models | Concepts                  | Key Components           | Common Implementations |
|------------------------|--------------------------|--------------------------|------------------------|
| Linear Regression      | parametric model, assume our data is linear, loss function |
| Decision Trees         | 
| Random Forests         | 
| Support Vector Machines| 
| Neural Networks        | 


## Structuring ML 

![image](https://github.com/dhirajmahato/Machine-Learning-Models/assets/33785298/2fab00d6-ea7b-43dd-9e33-9fae5a0f3446)

## ML Algos

![image](https://github.com/dhirajmahato/Machine-Learning-Models/assets/33785298/37e5490f-084c-4ca1-9314-95a2e800c968)


src:
https://github.com/nvmcr/DataScience_HandBook/tree/main/Machine_Learning
https://github.com/dhirajmahato/Machine-Learning-Notebooks



