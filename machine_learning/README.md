# Machine Learning

## Decision Tree 
A supervised learning algorithm that can be used for both classification and regression problems. It predicts the value or class of the target variable by learning decision rules inferred from the training dataset. Each split in a decision tree affects its overall accuracy, and it splits the nodes on all available input variables where the results produce increasingly homogeneous sub-nodes.

The algorithm used in the decision tree determines how the input variables are 'prioritised' at each split. The cost functions used in ID3 and C.4.5 uses entropy, where the input variables with the largest information gain are chosen as nodes. Alternatively, CART uses Gini impurity, where nodes are chosen based on minimal impurity. 

Entropy: \
![equationEntropy](https://latex.codecogs.com/png.download?E%28S%29%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bc%7D%20-%20p_%7Bi%7D%20log_%7B2%7D%28p_%7Bi%7D%29) \
where S is the current state, p_{i} is the percentage of class i in a node of state S.


