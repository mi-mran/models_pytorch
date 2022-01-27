# Machine Learning

## Decision Tree 
A supervised learning algorithm that can be used for both classification and regression problems. It predicts the value or class of the target variable by learning decision rules inferred from the training dataset. Each split in a decision tree affects its overall accuracy, and it splits the nodes on all available input variables where the results produce increasingly homogeneous sub-nodes.

The algorithm used in the decision tree determines how the input variables are 'prioritised' at each split. The cost functions used in ID3 and C.4.5 uses entropy, where the input variables with the largest information gain are chosen as nodes. Alternatively, CART uses Gini impurity, where nodes are chosen based on minimal impurity. 

Entropy is the level of randomness in the information present. A higher value of entropy results in lower overall useful information that can be drawn from the information: 

&nbsp;

![equationEntropy](https://latex.codecogs.com/png.latex?E%28S%29%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bc%7D%20-%20p_%7Bi%7D%20log_%7B2%7D%28p_%7Bi%7D%29) 

&nbsp;

where S is the current state, p_{i} is the percentage of class i in a node of state S.

Information gain is a measure of how well an attribute is able to separate training examples based on their target class. Naturally, an decrease in entropy results in an increase in information gain. Therefore a decision tree would strive towards the maximum information gain and minimum entropy for any given attribute. Information gain can be calculated by subtracting the weighted average of the child nodes of the new split from the entropy of the parent node. We can see an example in calculating information gain as follows:

At each new split, the decision tree must decide which attribute would give rise to the highest information gain. In the example dataset below, two attributes and the target output is given. This example shows the calculation of the information gain from the parent node, where the entropy is known to be 1.

&nbsp;

| Attr. 1 | Attr. 2 | Target |
| --- | --- | --- |
| True | True | A |
| True | False | A |
| False | True | B |
| True | False | B |

&nbsp;

The entropy at the parent node is given by: 

![equationParentEntropy](https://latex.codecogs.com/png.latex?E&space;=&space;-\sum&space;P_{A}log_{2}(P_{A})&space;&plus;&space;P_{B}log_{2}(P_{B})) \
 This further simplifies to:
 
![equationParentEntropyNumerical](https://latex.codecogs.com/png.latex?E&space;=&space;-\sum&space;0.5log_{2}(0.5)&space;&plus;&space;0.5log_{2}(0.5)&space;=&space;1) 

&nbsp;

From the parent node, splits based on each attribute is now tested. If the information gain of the attribute is the greatest, the attribute is chosen as the split.

Taking Attribute 1 as the first split to be tested, the left split would contain "AAB" and the right split would contain "B".

&nbsp;

```
LEFT SPLIT - ATTR. 1
P(A) = 2/3
P(B) = 1/3
```

Entropy of Attr. 1 (Left Split):

![equationEntropyAttr1Left](https://latex.codecogs.com/png.latex?E_{Left,Attr.1}&space;=&space;-\sum&space;\frac{2}{3}log_{2}(\frac{2}{3})&space;&plus;&space;\frac{1}{3}log_{2}(\frac{1}{3})&space;=&space;0.9) 

```
RIGHT SPLIT - ATTR. 1
P(B) = 1/1
```

Entropy of Attr. 1 (Right Split): <br />

![equationEntropyAttr1Right](https://latex.codecogs.com/png.latex?E_{Right,Attr.1}&space;=&space;-\sum&space;1log_{2}(1)&space;=&space;0)

Information gain can be calculated as:

```
I. Gain = E(parent) - weighted ave. of children nodes
```

Therefore the information gain of Attribute 1 is:

![equationInformationGainCalc](https://latex.codecogs.com/png.latex?IG(Attr.1)&space;=&space;1&space;-&space;(\frac{3}{4}*0.9&space;&plus;&space;\frac{1}{4}*0)&space;=&space;0.325)


Similarly, the information gain of Attribute 2 is 0. Hence we can conclude that Attribute 1 has the highest information gain and would form the first split of the decision tree.

**NOTE: This implementation only considers classification problems.**

---

## K Nearest Neighbours
Another supervised learning algorithm that helps us compute a predicted class or continuous value based on the labels of its K nearest neighbours. The distance between the neighbours determines the final computation of the predicted output value.

When given an unseen data point, the algorithm first identifies the K points that are closest to the data point based on the feature space. The distance between the data point and each K point can be calculated using various metrics, such as Manhattan, Euclidean and Hamming distances.

In a classification scenario, the class that dominates within the K points is selected as the output class. In a regression scenario, the average of all the K points is computed as the output value.

Although K nearest neighbours is a simple, interpretable and highly versatile algorithm, it does have several drawbacks. With an increasing volume of data, the the prediction step may be slowed. In addition, significant memory is used to store the training data. Lastly, the algorithm is sensitive to the scale of the data and the presence of irrelevant features.

---

## Linear Regression
Predicting a continuous value output, linear regression is another machine learning algorithm that allows for straightforward interpretation. 

When working with a single input variable, the relationship between the input and output can be described as:

![equationSingleLinearRegression](https://latex.codecogs.com/png.image?\dpi{110}&space;\hat{y}&space;=&space;m_{0}X&space;&plus;&space;C) 

where y-hat is the predicted target value, X is the input variable, and m<sub>0</sub> & C are parameters of the model to be tuned to the data distribution. The 

In order to tune the model, a cost function can be applied, where the predicted value and the actual value are compared. Optimisation of the unknown parameters can be done using ordinary least squares (OLS) or gradient descent. There are various cost functions that can be applied, such as mean absolute error (MAE), mean squared error (MSE), root mean squared error (RMSE), mean absolute percentage error (MAPE) and mean percentage error (MPE):

MAE:

![equationMAE](https://latex.codecogs.com/png.image?\dpi{110}&space;MAE&space;=&space;\frac{1}{N}&space;\sum_{i=1}^{n}&space;\left|&space;y_{i}&space;-&space;\hat{y}_{i}&space;\right|)

where y<sub>i</sub> is the actual value and N is the total number of samples in the given dataset.

MSE:

![equationMSE](https://latex.codecogs.com/png.image?\dpi{110}&space;MSE&space;=&space;\frac{1}{N}&space;\sum_{i=1}^{n}&space;(y_{i}&space;-&space;\hat{y}_{i})^{2})

RMSE:

![equationRMSE](https://latex.codecogs.com/png.image?\dpi{110}&space;RMSE&space;=&space;\sqrt{\frac{1}{N}&space;\sum_{i=1}^{n}&space;(y_{i}&space;-&space;\hat{y}_{i})^{2})

MAPE:

![equationMAPE](https://latex.codecogs.com/png.image?\dpi{110}&space;MAPE&space;=&space;\frac{100}{N}&space;\sum_{i=1}^{n}&space;\left|&space;\frac{y_{i}&space;-&space;\hat{y}_{i}}&space;{y_{i}}\right|)

MPE:

![equationMPE](https://latex.codecogs.com/png.image?\dpi{110}&space;MPE&space;=&space;\frac{100}{N}&space;\sum_{i=1}^{n}&space;\frac{y_{i}&space;-&space;\hat{y}_{i}}&space;{y_{i}})

In the case where more than one input variable is considered, a multiple linear regression model can be used, which is extended as the form:

![equationMultipleLinearRegression](https://latex.codecogs.com/png.image?\dpi{110}&space;\hat{y}&space;=&space;m_{0}X_{0}&space;&plus;m_{1}X_{1}&space;&plus;m_{2}X_{2}&space;&plus;&space;C)

where m<sub>n</sub> represents the coefficient of the independent variable n and X<sub>n</sub> represents the respective independent variables.

However, when using linear regression, several assumptions are made:

1. There is a linear relationship between the input variable and the output variable. Inspecting their scatter plots would provide a visual check for such a relationship.

2. The variance of the residual should be the same throughout the data points (residual refers to the difference between the expected output value and the predicted output value). This would also mean that the data is homoscedastic. The homoscedasticity can be validated using various tests such as the Breusch-Pagan and Goldfeld-Quandt tests.

3. The input variables should not be highly correlated with one another. If there is such high correlation, there would be an issue in identifying the specific input variable that affects the variance in the output variable. This can be verified through the Variance Inflation Factor (VIF).

4. Autocorrelation should also be minimised. If the values of residuals are not independent, the accuracy of the model is reduced and there is an underestimation of the standard error. Autocorrelation can be tested using the Durbin-Watson test.

5. Lastly, the residuals must be normally distributed. This can be verified through a goodness of fit test, such as the Kolmogorov-Smirnov or Shapiro-Wilk tests. If the data is not normally distributed, it could possibly be remedied by a non-linear transformation (eg. log transform).

