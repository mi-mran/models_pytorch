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