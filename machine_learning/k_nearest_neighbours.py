from math import dist
import numpy as np

def euclidean_dist(a, b):
    return np.sqrt(np.sum(np.square(a - b)))

def manhattan_dist(a, b):
    return sum(abs(x1 - x2) for x1, x2 in zip(a, b))

def hamming_dist(a, b):
    return np.count_nonzero(a != b)

class KNN:
    def __init__(self, k=3, out_type='categorical', dist_type='euclidean'):
        # out_type defines whether the output is categorical or a continuous value
        # dist_type defines distance metric that is used

        self.k = k
        self.out_type = out_type
        self.dist_type = dist_type

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # input X is a list

        predictions = [self._predict(x) for x in X]

        return np.array(predictions)

    def _predict(self, x):
        # compute distance depending on distance metric chosen

        dist = self.dist_type

        if dist == 'euclidean':
            distances = [euclidean_dist(x, x_train) for x_train in self.X_train]
        elif dist == 'manhattan':
            distances = [manhattan_dist(x, x_train) for x_train in self.X_train]
        elif dist == 'hamming':
            distances = [hamming_dist(x, x_train) for x_train in self.X_train]

        k_idx = np.argsort(distances)[:self.k]

        k_nearest_labels = [self.y_train[i] for i in k_idx]

        if self.out_type == 'categorical':
            counter = {}

            for label in k_nearest_labels:
                if label not in counter:
                    counter[label] = 0

                counter[label] += 1

            most_common = max(counter, key=lambda key: counter[key])

            return most_common

        elif self.out_type == 'continuous':
            # get the respective distances from the unknown data point
            # sort to get the k nearest neighbours
            # calculate the average of nearest labels

            return np.mean(k_nearest_labels)

if __name__ == '__main__':
    # testing with sklearn toy dataset
    # iris dataset consists of 4 input features, all of which are continuous values, and 3 output classes (Iris-Setosa, Iris-Versicolour, Iris-Virginica)
    
    import sklearn.datasets
    from sklearn.model_selection import train_test_split
    from utils import accuracy

    raw_data = sklearn.datasets.load_iris()

    X = raw_data.data
    y = raw_data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    clf = KNN(k=3, out_type='categorical', )
    clf.fit(X_train, y_train)

    y_preds = clf.predict(X_test)

    # for classification accuracy
    print("Accuracy: ", accuracy(y_test, y_preds))