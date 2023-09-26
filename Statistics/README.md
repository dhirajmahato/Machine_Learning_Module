## Statistics

   
### A. Introduction

#### 1. Population and Sample
- **Population**: Population is the set of all possible observations or records (often called cases, subjects, or data points) for a given context of the problem.
- **Sample**: A sample is a subset of data from a larger data set called population. eg: the opinion polls are based on opinions expressed by a subset of voters called sample.

In many real-life problems, we make inferences about the population based on sample data or based on multiple samples for cross validation.
There are many challenges in sampling (process of selecting a sample from the population). An incorrect sample may result in bias and incorrect inference about the population.

- *Random Sampling* is a process in which each member in population has equal chance of being chosen for sample.

#### 2. Variable Data Types, Classification, Scales
- Data can be classified into 3 different types of data:
   1. **Cross-Sectional Data:** collected on several variables at the same point in time.   
   2. **Time Series Data:** collected on a single variable, such as demand for smartphones over several time intervals (weekly, monthly etc).
   3. **Panel or Longitudinal Data:** collected on several variables (multiple dimensions) over several time intervals. such as gross domestic product (GDP), Gini index, and unemployment rate collected forseveral countries over several years.
- observational data and experimental data (for establishing causal relationship). 
- Structured (Matrix) or Unstructured (audio, image ..)
- Structured data may contain variables that are either numeric or alpha numeric and may follow different scales of measurement
- Different measurement scales of variables like
   - **Nominal scale** (*Qualitative*)  That is, basic mathematical operations are meaningless  ---> Categorical variables marital status (single, married, divorced) and
industry type (manufacturing, healthcare, banking, insurance, and finance)
   - Any variable for which the ratios can be computed and are meaningful is called a **ratio scale** variable. Many continuous variables come under ratio scale; for example: demand for a product, market share of a brand, sales, salary, and so on.  
   - **Ordinal scale** is a variable type in which the value of the variable is captured from a rank ordered set. eg: g 5-point Likert scale in which 1 = Poor, 2 = Fair, 3 = Good, 4 = Very Good, and 5 = Excellent.
   - An **Interval scale** corresponds to a variable type in which the value of the variable is chosen from an interval set. In an interval scale, the ratios do not make sense.  The reference point is fixed arbitrarily. eg: Variables such as temperature measured in centigrade (oC) and intelligence quotient (IQ) score are examples of interval scales.


#### 3. Measures of Central Tendency

Central tendency is a statistical measure that represents the center or average value of a dataset. Mean, median and mode are the three measures of central tendency
1. **Mean** - arithmetic average value of the data, an important property of mean is that the summation of deviation of all records from the mean is zero.
   - Making decisions solely based on mean value is not advisable since variability in the data can be more critical than the mean.
   - Another issue with mean is that it is affected significantly by outliers.
2. **Median** - Median is the value that divides the data into two equal parts.
   - Median is much more stable than the mean value, as adding a new observation may not change the median significantly.
   - drawback of median is that it is not calculated using the entire dataset like in the case of mean.
3. **Mode** - the most frequently occurring value in the dataset.
   - Mode is the only measure of central tendency which is valid for qualitative (nominal) variables, since the measures mean and median for nominal scale variables are meaningless.

- If the data is symmetrically distributed then Mean = Median = Mode
- If the distribution is Skewed, then Median is the best measure of central tendency Mean is most sensitive for skewed data.
- Mode is the best measure for categorical data

#### 4. Measures of Variation/Dispersion

Dispersion or variability describes how items are distributed from each other and the centre of a distribution. <br/> Measures of variability are useful in identifying how close the records are to the mean value and outliers in the data. 

Predictive analytics techniques such as regression attempt to explain variation in the outcome variable (Y) using predictor variables (X).
If a variable or feature has very low variability, then it is unlikely to have a statistically significant relationship with an outcome variable.

There are 4 methods to measure the dispersion of the data:
- Range: difference between the maximum and minimum values of the data in the sample, captures the data spread.
   - conversion of a continuous number into buckets is a part of feature engineering and may improve the model        
- Inter Quartile Distance(IQD/IQR): distance between Quartile 1 (Q1) and Quartile 3 (Q3) in a data set.Values of data below Q1 – 1.5 IQD and above Q3 + 1.5 IQD are classified as potential outliers.
   -  its use appropriate only in the case of univariate data (data with one dimension). In the case of multivariate data, we use distance measures such as
Mahalanobis distance and Cook’s distance to identify outliers.
- Mean Absolute Deviation (MAD): the average deviation from the mean value of the data. MAD is always finite and defined
- Variance: 
- Standard Deviation (Best Measure)
- Coefficient of Variation


  
#### Types
1. Descriptive statistics are used to summarize and describe the characteristics of a dataset. 
2. Predictive
3. Prescriptive
4. Inferential statistics are used to draw conclusions about a population based on a sample of data. Techniques like hypothesis testing and confidence intervals are examples of inferential statistics.

### D. Statistics Distribution
- Data distribution is the **frequency distribution** of individual values in a data set.
  
- **Probability distributions** describe the likelihood of different outcomes in a random event.
   - Discrete distributions are used when the variable can take on a countable number of distinct values, such as the binomial distribution for the number of successes in a fixed number of trials. 
   - Continuous distributions are used when the variable can take on any value within a range, such as the normal distribution for variables like height or weight.
- **Normal Distribution**
   - The normal distribution (or Gaussian distribution) is a continuous probability distribution that is symmetric and bell-shaped.
   - It is characterized by its mean (μ) and standard deviation (σ), which determine the location and spread of the distribution, respectively.
   - Example: The heights of adult males in a population often follow a normal distribution. If the mean height is 175 cm and the standard deviation is 6 cm, you can use the normal distribution to calculate the probability of finding a male with a height between certain values or above a  certain height.
- **Binomial distribution** describes the probability of obtaining a specific number of successes in a fixed number of independent trials, each with the same probability of success.
   - It is characterized by two parameters: the number of trials (n) and the probability of success (p).
   - Example: Tossing a fair coin h 10 times can be modeled using a binomial distribution. The probability of getting exactly 5 heads (successes) in 10 coin flips can be calculated using the binomial distribution formula.
 
- **Poisson Distribution**
- **Exponential Distribution**

### E. Statistical Inference
#### 1. Hypothesis Testing (Null and Alternative Hypotheses):
Hypothesis testing is a statistical method used to make inferences about a population based on sample data.

Common assumptions include:
- Independence: The observations in the sample are independent of each other.
- Random sampling: The sample is representative of the population being studied.
- Normality: The data or test statistics follow a normal distribution.
- Homogeneity a of variance: The variance of the variables is equal across groups being compared.

#### 2. Type I and Type II Errors:
#### 3. Confidence Intervals:
#### 4. P-values:
