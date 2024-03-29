#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold

class Evaluation:
    """This class provides functions for evaluating classifiers """
        
    def generate_cv_pairs(self, n_samples, y, n_folds=5, n_rep=1):
        """ Train and test pairs according to statified k-fold cross validation with ranodmization 

        Parameters
        ----------
        n_samples : int
            The number of samples in the dataset

        y : array-like, shape (n_sapmles), 
            The labels of the data.

        n_folds : int, optional (default: 5)
            The number of folds for the cross validation

        n_rep : int, optional (default: 1)
            The number of repetions for the cross validation

        Returns
        -------
        cv_splits : list of tupels, each tuple contains two arrays with indices
            The first array corresponds to the training data, the second to the
            testing data for the current split. The list has the length of
            *n_folds* x *n_rep*.

        """
        cv_splits = []
        for i in range(n_rep):
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=i)
            cv_splits.extend(list(cv.split(np.zeros(n_samples),y)))
        return cv_splits
    
    def apply_cv(self, X, y, train_test_pairs, classifier, **kwargs):
        """ Evaluate classifier on testing data and return confusion matrix 
        
        Parameters
        ----------
        X : array-like, shape (n_samples, feature_dim)
            All data used within the cross validation
        
        y : array-like, shape (n_samples)
            The actual labels for the samples
        
        train_test_pairs : list of tupels, each tuple contains two arrays with indices
            The first array corresponds to the training data, the second to the
            testing data for the current split
        
        classifier : function
            Function that trains a classifier and returns the predictions for 
            the testing data. Arguments of the functions are the training
            data, the correct labels for the training data, the testing data.
            Further, keyword arguments given to *apply_cv* are passed to this
            function.
        
        Returns
        -------
        confusion_matrix : array-like, shape (n_classes, n_classes)
            The averaged confusion matrix (rows actual values, columns predicted
            values) across all test splits.
        """
        
        n_labels = np.unique(y).size
        confusion_matrix = np.zeros((n_labels, n_labels))
        n = len(train_test_pairs)
        for train_index, test_index in train_test_pairs:
            predictions = classifier(X[train_index,:], y[train_index], 
                X[test_index,:], **kwargs)
            confusion_matrix += self.confusion_matrix(predictions, y[test_index])
        confusion_matrix /= float(n)
        return confusion_matrix
    
    def confusion_matrix(self, predictions, labels):
        """ Return normalized confusion matrix 
        
        Computes the confusion matrix and normalizes it, so that all values sum
        up to one.
        
        Parameters
        ----------
        predictions : array-like, shape (n_samples)
            The classification outcome
        
        labels : array-like, shape (n_samples)
            The actual labels for the samples
        
        Returns
        -------
        confusion_matrix : array-like, shape (n_classes, n_classes)
            A normalized confusion matrix, where all entries sum up to 1.
        """
        #Prog for confusion matrix
        
        K = len(np.unique(labels)) 
        result = np.zeros((K, K))
        for i in range(len(labels)):
            result[labels[i]][predictions[i]] += 1
        sum_horizontal = np.sum(result, axis=1)
        return result.astype('float') / result.sum(axis=1)[:, np.newaxis]
        
class LearningAlgorithms:
    """ Algorithms for supervised learning """
    
    def decision_tree(self, X_train, y_train, X_test, **kwargs):
        """ Train a decision tree and return test predictions 
        
        Parameters
        ----------
        X_train : array-like, shape (n_train-samples, feature_dim)
            The data used for training 
        
        y_train : array-like, shape (n_train-samples)
            The actual labels for the training data
        
        X_test : array-like, shape (n_test-samples, feature_dim)
            The data used for testing
        
        Returns
        -------
        predictions : array-like, shape (n_test-samples)
            The classification outcome
        
        """
        c = DecisionTreeClassifier(**kwargs)
        c.fit(X_train, y_train)
        return c.predict(X_test)
    
    def bagging(self, X_train, y_train, X_test, bag_size=None, 
            num_bags=10, base_learner=DecisionTreeClassifier, **kwargs):
        """ Build Bagging model on training data and return test predictions 
            
        ### THE SUMMARY OF YOUR ALGORITHM GOES HERE ###
            
        Parameters
        ----------
        X_train : array-like, shape (n_train-samples, feature_dim)
            The data used for training 
        
        y_train : array-like, shape (n_train-samples)
            The actual labels for the training data

        X_test : array-like, shape (n_test-samples, feature_dim)
            The data used for testing
        
        bag_size : int or None, optional (default: None)
            Number of instances used for training of an ensemble member. If
            None *bag_size* is set to n_train-samples.

        num_bags : int, optional (default: 10)
            Number of bags and hence number of ensemble learners.
            
        base_learner : class, optional (default: DecisionTreeClassifier)
            Scikit-learn classifier. Keyword arguments are passed to the class
            if an instance is created.
            
        Returns
        -------
        predictions : array-like, shape (n_test-samples)
            The classification outcome
        
        """
        #Getting Bag size 
        def get_bag_size_train(X_train_s, y_train_s, bag_s):
            random_entities = np.random.randint(0, X_train_s.shape[0], bag_s)
            return X_train_s[random_entities].copy(), y_train_s[random_entities].copy()
        
        #Creates N classifiers of type base_learner and trains them each on the given X_train, y_train
        #data. Tests each of the classifiers based on the X_test data. The classifiers' output is taken as
        #the majority voting for each sample.
        
        all_classifiers = []
        for i in range(num_bags) :
            all_classifiers.append(base_learner(**kwargs))
            if bag_size:
                X_train_now, y_train_now = get_bag_size_train(X_train, y_train, bag_size)
                all_classifiers[i].fit(X_train_now, y_train_now)
            else:
                all_classifiers[i].fit(X_train, y_train)
        predictions = np.zeros((X_test.shape[0], 2))
        for i in range(num_bags):
            current_preds = all_classifiers[i].predict(X_test)
            for i in range(current_preds.shape[0]):
                predictions[i][current_preds[i]] += 1
        voted_preds = []
        for i in range(predictions.shape[0]):
            if predictions[i][0] > predictions[i][1]:
                voted_preds.append(0)
            else:
                voted_preds.append(1);
        return np.array(voted_preds, dtype=int)
    
    
def question_a(X, y, a, c):
    y = y.astype(int)
    X_train = X[:-30]
    y_train = y[:-30]
    X_test = X[-30:]
    y_test = y[-30:]
    preds = a.decision_tree(X_train, y_train, X_test)
    confusion_matrix = c.confusion_matrix(preds, y_test)
    print(confusion_matrix)



def question_b(X, y, a, c):
    y = y.astype(int)
    cv_splits = c.generate_cv_pairs(y.shape[0], y)
    print(c.apply_cv(X, y, cv_splits, a.bagging))

def question_c(X, y, a, c):
    y = y.astype(int)
    num_elements = int(X.shape[0] / 5) *-1
    X_train = X[:num_elements]
    y_train = y[:num_elements]
    X_test = X[num_elements:]
    y_test = y[num_elements:]
    preds = a.decision_tree(X_train, y_train, X_test)
    confusion_matrix = c.confusion_matrix(preds, y_test)
    print(confusion_matrix)
    
def question_d(X, y, a, c, bag_size):
    y = y.astype(int)
    cv_splits = c.generate_cv_pairs(y.shape[0], y)
    print(c.apply_cv(X, y, cv_splits, a.bagging, bag_size=bag_size))
    
if __name__ == '__main__':
    
    # load diabetis dataset
    data = np.loadtxt('diabetes_data.csv', delimiter=',')
    X, y = data[:,1:], data[:,0]
    
    c = Evaluation()
    a = LearningAlgorithms()
    
    print("Question_a confusion matrix:")
    question1(X.copy(), y.copy(), a, c)
    
    print()
    print("Question_b Confusion matrix")
    question2(X.copy(), y.copy(), a, c)
    
    print()
    print("Question_c Confusion matrix:")
    question3(X.copy(), y.copy(), a, c)
    
    print()
    print("Question_d Confusion matrix:")
    question4(X.copy(), y.copy(), a, c, 300)
