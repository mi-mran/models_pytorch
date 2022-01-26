# THIS IMPLEMENTATION CURRENTLY ONLY CONSIDERS DECISION TREE CLASSIFICATION PROBLEMS

import numpy as np

def entropy(y):
    # counts no. of occurrences of each class label
    hist = np.bincount(y)
    # calculate P(class) for each class, P(A) = no. of A / len. of y
    p_classes = hist / len(y)

    return -np.sum([p * np.log2(p) for p in p_classes if p > 0])

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)

    return accuracy

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        # value represents label on a leaf node
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        if self.value is None:
            return False
        return True


class DecisionTree:
    def __init__(self, min_sample_split=2, max_depth=100, n_feats=None):
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, x, y):
        # ensure that the number of features considered cannot be greater than the number of features present in the x matrix
        # x is a 2D numpy array, second dimension represents number of features
        self.n_feats = x.shape[1] if not self.n_feats else min(self.n_feats, x.shape[1])

        # for training the tree
        self.root = self._grow_tree(x, y)

    
    def _grow_tree(self, x, y, depth=0):
        n_samples, n_features = x.shape
        n_labels = len(np.unique(y))

        # stopping criteria: max depth, min. samples at a node, no more class distribution at a node
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_sample_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # greedy search
        best_feat, best_thresh = self._best_criteria(x, y, feat_idxs)
        left_idxs, right_idxs = self._split(x[:, best_feat], best_thresh)
        
        left = self._grow_tree(x[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(x[right_idxs, :], y[right_idxs], depth+1)

        return Node(best_feat, best_thresh, left, right)


    def _most_common_label(self, y):
        # intentionally choosing not to use Python's Counter function here
        counter = {}

        for label in y:
            if label not in counter:
                counter[label] = 0
            
            counter[label] += 1
        
        most_common = max(counter, key=lambda x: counter[x])

        return most_common

    def _best_criteria(self, x, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            x_col = x[:, feat_idx]
            thresholds = np.unique(x_col)

            for threshold in thresholds:
                gain = self._information_gain(y, x_col, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    def _information_gain(self, y, x_col, threshold):
        # information gain = entropy of parent - weighted average of entropy of children
        parent_entropy = entropy(y)
        left_idxs, right_idxs = self._split(x_col, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n_samples_parent = len(y)
        n_samples_left, n_samples_right = len(left_idxs), len(right_idxs)
        entropy_left, entropy_right = entropy(y[left_idxs]), entropy(y[right_idxs])

        weighted_child_entropy = (n_samples_left / n_samples_parent) * entropy_left + (n_samples_right / n_samples_parent) * entropy_right

        information_gain = parent_entropy - weighted_child_entropy

        return information_gain

    def _split(self, x_col, threshold):
        left_idxs = np.argwhere(x_col <= threshold).flatten()
        right_idxs = np.argwhere(x_col > threshold).flatten()

        return left_idxs, right_idxs

    def predict(self, X):
        # traverse trained tree to output predicted label/value
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        # check if we have reached a leaf node

        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        
        return self._traverse_tree(x, node.right)

if __name__ == '__main__':
    # testing with sklearn toy dataset
    # consists of 30 input features and two output classes (malignant & benign)

    import sklearn.datasets
    from sklearn.model_selection import train_test_split
    
    raw_data = sklearn.datasets.load_breast_cancer()

    X = raw_data.data
    y = raw_data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    clf = DecisionTree(max_depth=10)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy(y_test, y_pred)

    print("Accuracy: ", acc)