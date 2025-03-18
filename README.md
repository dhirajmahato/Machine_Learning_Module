Here, I will add Machine Learning Models and usage and Explanations

## Types of ML
based on Human supervision:
1. Supervised
2. Unsupervised
3. Ensemble Learning
4. Reinforcement
5. Neural Networks

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

Other techniques
1. Ensemble Learning Algorithms: Bagging, Boosting, stacking
2. Deep Learning Algorithms: CNN, RNN
3. Bayesian Learning Algorithms : Naive Bayes, Bayesian Networks
4. Instance-Based Learning Algorithms: k-Nearest Neighbors (k-NN), Locally Weighted Regression (LWR)
5. Clustering Algorithms : K-Means, Hierarchical clustering
6. Dimensionality Reduction Algorithms: PCA (linear), t-SNE(non linear)


### Concepts

**Maximumm Liklihood Estimation (MLE)** is a method that determines values of the parameters of a model such that they maximise the likelihood of observed data given a probability distribution.


## A. Supervised ML Table
### Regression
Here, the model predicts the relationship between input features (independent variables) and a continuous output variable.
| Machine Learning Models | Concepts                  |      Usecases           |
|------------------------|--------------------------|---------------------|
| [Linear Regression](https://nbviewer.org/github/maykulkarni/Machine-Learning-Notebooks/blob/master/02.%20Regression/1A.%20Linear%20Regression%20and%20Gradient%20Descent%28Theory%29.ipynb)     | There are four assumptions: <br /> 1. **Linearity:** The relationship between X and the mean of Y is linear. $Y=\beta_{0}+\beta{1}X+\epsilon\text{(Error term)}$ <br/> Detection: Residual plots (against X), nicely and event spread. <br /> 2. **Homoscedasticity:** the variance of error terms are similar across the values of the independent variables. A plot of standardized residuals versus predicted values can show whether points are equally distributed across all values of the independent variables.<br /> 3. **Little to no Multicollinearity:** Independent variables are not highly correlated with each other. This assumption is tested using Variance Inflation Factor (VIF) values. One way to deal with multicollinearity is subtracting mean.<br /> 4. **Normality:** Residuals should be normally distributed. This can be checked using histogram of residuals. <br /> <br/><br/> - Feature scaling is required <br /> - Sensitive to missing value| Good for sparse, high-dimensional data <br /> 1. Advance House Price Prediction <br /> 2. Flight Price Prediction |
| Polynomial regression  |more the degree of the polynomial the better is the fit but the more is the issue of overfitting.  |    |


**Evaluation Metrics**: 
Regression models are typically evaluated using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²), among others.

### Classification:

| Machine Learning Models | Concepts                  |   Usecases            |
|------------------------|--------------------------|------------------------|
| Logistics Regression   | Assumptions: <br/> 1. The outcome is binary<br/> 2. Linear relationship between the logit of the outcome and the predictors. <br/> Logit function: $\text{logit}(p) = \log\left(\frac{p}{1-p}\right)$ <br/> $\(p\)$: probability of the outcome<br/> 3. No outliers/extreme values in the continuous predictors<br/> 4. No multicollinearity among the predictors <br/> <br/> Sigmoid/Logistic Function: S-shaped curve that takes any real number and maps it between 0 and 1 $f(x)=\frac{1}{1+e^{-x}}$ <br/> <br/> - Feature scaling is required <br /> - Sensitive to missing value |
| Decision Tree          | Pros: <br/> 1. Handles categorical data, missing valuses <br/> 2. normalization not needed <br/> 3. can heandle non linear relationship <br/> Cons: <br/> 1. overfitting <br/> 2. need ensemble to improve accuracy |  
| Random Forest (Ensemble Learning)| 1. ensembling (putting many weak trees together) <br/> 2. bagging (parallel uncorelated trees or estimators) - to take sample of the dataset with replacement |
| Support Vector Machines| 
| Naive Bayes            |
| K-Nearest Neighbour    |

**Evaluation Metrics:**
Classification models are evaluated using metrics such as accuracy, precision, recall, F1-score, and the confusion matrix, depending on the problem and class distribution.

## B. Unsupervised ML Table
### Clustering
Here, the model predicts the relationship between input features (independent variables) and a continuous output variable.
| Machine Learning Models | Concepts                  |      Usecases           |
|------------------------|--------------------------|---------------------|
| K-Means CLustering     |



## Structuring ML 

![image](https://github.com/dhirajmahato/Machine-Learning-Models/assets/33785298/2fab00d6-ea7b-43dd-9e33-9fae5a0f3446)

## ML Algos

![image](https://github.com/dhirajmahato/Machine-Learning-Models/assets/33785298/37e5490f-084c-4ca1-9314-95a2e800c968)


src:
https://github.com/nvmcr/DataScience_HandBook/tree/main/Machine_Learning
https://github.com/dhirajmahato/Machine-Learning-Notebooks



