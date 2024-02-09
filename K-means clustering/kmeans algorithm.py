#!/usr/bin/env python
# coding: utf-8

# In[45]:


# !/usr/bin/env python3

# do not use any other imports!
import numpy as np
import matplotlib.pyplot as plt
import random


class Kmeans:
    def initialize_clusters(self,points, k):
        """Initializes clusters as k randomly selected points from points."""
        return points[np.random.randint(points.shape[0], size=k)]


# Function for calculating the distance between centroids
def get_distances(centroid, points):
        """Returns the distance the centroid is from each data point in points."""
        return np.linalg.norm(points - centroid, axis=1)

if __name__ == '__main__':
    # Load data
    X = np.genfromtxt('cluster_dataset2d.txt', delimiter=',')
    c = Kmeans()

    # In[46]:

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(X[:, 0], X[:, 1], alpha=0.5)
    plt.suptitle('Before clustering')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$');

    # In[47]:

    k = 3
    maxiter = 50

    # Initialize our centroids by picking random data points
    centroids = c.initialize_clusters(X,k)

    # Initialize the vectors in which we will store the
    # assigned classes of each data point and the
    # calculated distances from each centroid
    classes = np.zeros(X.shape[0], dtype=np.float64)
    distances = np.zeros([X.shape[0], k], dtype=np.float64)

    # Loop for the maximum number of iterations
    for i in range(maxiter):
        for i, c in enumerate(centroids):
            distances[:, i] = get_distances(c, X)
            classes = np.argmin(distances, axis=1)


    for c in range(k):
        centroids[c] = np.mean(X[classes == c], 0)



    group_colors = ['skyblue', 'coral', 'lightgreen']
    colors = [group_colors[j] for j in classes]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(X[:, 0], X[:, 1], color=colors, alpha=0.5)
    ax.scatter(centroids[:, 0], centroids[:, 1], color=['blue', 'darkred', 'green'], marker='o', lw=2)
    plt.suptitle('After clustering')
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    plt.show()






