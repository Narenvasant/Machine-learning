import numpy as np
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


class Classification:
    """This class holds different classification algorithms and the cv function
    """

    def apply_k_fold_cv(self, X, y, classifier, n_folds, neighbour_measure, distance_measure):
        """K fold cross validation.

        Parameters
        ----------
        X : array-like, shape (n_samples, feature_dim)
            The data for the cross validation

        y : array-like, shape (n-samples, label_dim)
            The labels of the data used in the cross validation

        classifier : function
            The function that is used for classification of the training data

        n_splits : int, optional (default: 5)
            The number of folds for the cross validation

        kwargs :
            Further parameters that get used e.g. by the classifier

        Returns
        -------
        accuracies : array, shape (n_splits,)
            Vector of classification accuracies for the n_splits folds.
        """
        assert X.shape[0] == y.shape[0]

        if len(X.shape) < 2:
            X = np.atleast_2d(X).T
        if len(y.shape) < 2:
            y = np.atleast_2d(y).T

        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = []

        for train_index, test_index in cv.split(X):
            train_data = X[train_index, :]
            train_label = y[train_index, :]
            test_data = X[test_index, :]
            test_label = y[test_index, :]

            score = classifier(train_data, test_data,
                               train_label, test_label, neighbour_measure, distance_measure)

            scores.append(score)

        return np.array(scores)

    def kNN_classifier(self, X_train, X_test, y_train, y_test,
                       neighbours, metric):
        """K nearest neighbor classifier.

        Parameters
        ----------
        X_train : array-like, shape (n_samples, feature_dim)
            The data for the training of the classifier

        X_test : array-like, shape (n_samples, feature_dim)
            The data for the test of the classifier

        y_train : array-like, shape (n-samples, label_dim)
            The labels for the training of the classifier

        y_test : array-like, shape (n-samples, label_dim)
            The labels for the test of the classifier

        neighbors : int, optional (default: 1)


        metric : function
            The function that is used as a metric for the kNN classifier

        Returns
        -------
        accuracy : double
            Accuracy of the correct classified test data
        """

        ### YOUR IMPLEMENTATION GOES HERE ###

        predictedLabel = np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            dists = np.zeros(X_train.shape[0])
            for j in range(X_train.shape[0]):
                dists[j] = metric(X_test[i, :], X_train[j, :], X_train)
            # print(dists)
            nn = np.argsort(dists)[:neighbours]
            # print(nn)
            lbl = y_train[nn]
            indLbl, occs = np.unique(lbl, return_counts=True)
            predictedLabel[i] = indLbl[np.argmax(occs)]
        # print(predictedLabel)

        y_test1 = np.concatenate(y_test)
        # print(y_test1)

        correct = np.sum(predictedLabel == y_test1)
        # print(correct)
        # print(len(y_test))
        accuracy = float((correct) / len(y_test))
        return (accuracy)

    def normalized_euclidean_distance(self, data_a, data_b, X_data):
        """Normalized euclidean distance metric"""
        ### YOUR IMPLEMENTATION GOES HERE ###

        sum1 = 0

        for i, (e1, e2) in enumerate(zip(data_a, data_b)):
            sum1 += (((e1 - e2) ** 2) / (np.std(X_data[:, i])))
        # print(np.sqrt(sum1))
        return (np.sqrt(sum1))

    def manhatten_distance(self, data_a, data_b, X_data):
        """Distance metric of your choice"""
        ### YOUR IMPLEMENTATION GOES HERE ###

        sum1 = np.sum(np.abs(data_a - data_b))
        # print(sum1)
        return (sum1)

    def chebyshev_distance(self, data_a, data_b, X_data):
        """Distance metric of your choice"""

        ### YOUR IMPLEMENTATION GOES HERE ###
        # print(data_a,data_b)
        sum1 = np.max(np.abs(data_a - data_b))
        # print(sum1)
        return (sum1)


if __name__ == '__main__':

    # Instance of the Classification class holding the distance metrics and
    # classification algorithm
    iris = load_iris()

    ### YOUR IMPLEMENTATION FOR EXERCISE GOES HERE ###

    print("Normalised Euclidean distance")

    c1 = Classification()

    ans1 = c1.apply_k_fold_cv(iris.data, iris.target, c1.kNN_classifier, 10, 3, c1.normalized_euclidean_distance)
    print(ans1)

    print("Manhatten distance")
    c2 = Classification()
    ans2 = c2.apply_k_fold_cv(iris.data, iris.target, c2.kNN_classifier, 10, 3, c2.manhatten_distance)
    print(ans2)

    print("Chebyshev distance")
    c3 = Classification()
    ans3 = c3.apply_k_fold_cv(iris.data, iris.target, c3.kNN_classifier, 10, 3, c3.chebyshev_distance)
    print(ans3)

    print("For different values of k")
    c4 = Classification()
    fig, ax = plt.subplots(figsize=(8, 8))

    print("Iteration \t Normalised Euclidean dist \t Manhatten dist \t Chebyshev dist")
    ans4_mean=[]
    ans5_mean=[]
    ans6_mean=[]

    for k in range(100):
        # print("K=",k+1)
        ans4 = c4.apply_k_fold_cv(iris.data, iris.target, c4.kNN_classifier, 5, k + 1, c4.normalized_euclidean_distance)
        #print("finished euclidean")
        ans5 = c4.apply_k_fold_cv(iris.data, iris.target, c4.kNN_classifier, 5, k + 1, c4.manhatten_distance)
        #print("finished manhatten")
        ans6 = c4.apply_k_fold_cv(iris.data, iris.target, c4.kNN_classifier, 5, k + 1, c4.chebyshev_distance)

        # print(np.mean(ans4))
        ans4_mean.append(np.mean(ans4))
        ans5_mean.append(np.mean(ans5))
        ans6_mean.append(np.mean(ans6))

        print(k+1," \t ",np.mean(ans4)," \t ",np.mean(ans5)," \t ",np.mean(ans6))

    K=np.arange(1,101,1)

    ax.plot(K, ans4_mean,'o-',label='Normalised Euclidean dist')
    #ax.plot(K, ans5_mean,'o-',label='Manhatten dist')
    #ax.plot(K, ans6_mean,'o-',label='Chebyshev dist')
    ax.set_xlabel('$Value_K$')
    ax.set_ylabel('$Accuracy$')
    #ax.title('Normalised Euclidean Distance')

    ax.grid(True)
    plt.legend(loc='lower right')
    fig.suptitle('Accuracy vs K-neighbours')
    plt.show()