import numpy as np
import matplotlib.pyplot as plt
import random


def takefirst(array):
    return (array[1])


def sum_series(n):
    if n < 1:
        return (0)
    else:
        return (n + sum_series(n - 1))


def init_clusters(points, k):
    """Initializes clusters as k randomly selected points from points."""
    return points[np.random.randint(points.shape[0], size=k)]


# Function for calculating the distance between centroids
def get_distances(centroid, points):
    """Returns the distance the centroid is from each data point in points."""
    return np.linalg.norm(points - centroid, axis=1)


if __name__ == '__main__':
    # Load data
    X = np.genfromtxt('cluster_dataset2d.txt', delimiter=',')
    # X=  np.array([[2, 10], [2, 5], [8, 4], [5, 8], [7, 5], [6, 4], [1, 2], [4, 9]])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(X[:, 0], X[:, 1], alpha=0.5)
    plt.suptitle('Input Plot')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$');
    # print(X)
    k = [2, 3, 4, 5, 6, 7, 8, 9]
    maxiter = 50
    minimum_cindex_values = []
    average_cindex_values = []
    # print(np.shape(X)[0])
    for z in k:
        centroids = init_clusters(X, z)
        classes = np.zeros(X.shape[0], dtype=np.float64)
        distances = np.zeros([X.shape[0], z], dtype=np.float64)
        # print(z,"k value",centroids,"centroid")
        Scl = 0
        Cindex_average = []
        for j in range(maxiter):
            # Assign all points to the nearest centroid
            for i, c in enumerate(centroids):
                distances[:, i] = get_distances(c, X)
                # print(X)
            ans1 = np.zeros([X.shape[0], 3])

            # print(np.count_nonzero(distances))

            # class membership of each point by picking the closest centroid
            classes = np.argmin(distances, axis=1)
            # print(classes)

            # Calculating the C-index
            # print(classes)

            ##Splitting the array into classes
            c1, class_count = np.unique(classes, return_counts=True)
            # print(c1,class_count)

            for i, (classes1, data) in enumerate(zip(classes, X)):
                ans = list(np.append(classes1, data))
                # print(ans)
                ans1[i, :] = ans

            sorted_ans = ans1[np.argsort(ans1[:, 0])]
            # print(sorted_ans)

            c2, indices, class_count2 = np.unique(sorted_ans[:, 0], return_counts=True, return_index=True)
            sorted_split = np.split(sorted_ans, indices[1:])
            # print(c2)

            dist_sum = []
            N = []
            Scl = 0
            N_value = 0

            for i in c2:  # no. of classes
                n = class_count2[int(i)]
                array1 = np.asarray(sorted_split[int(i)])
                array2 = np.vstack(array1[:, 1:])

                distance = []
                number = []

                ##Calculation for Scl
                for a in range(class_count2[int(i)]):
                    centre = array2[a, :]
                    point = array2[a:, :]
                    distance.append(np.sum(get_distances(centre, point)))
                    number.append(np.count_nonzero(get_distances(centre, point)))
                    # print(point,"point")

                dist_sum.append(np.sum(distance))

                ##Calculation for N
                N = np.append(N, number)

            N_value = int(np.sum(N))
            # print("N",N_value)

            # Printing Scl value
            # print(distance_array,"distance")
            Scl = np.sum(dist_sum)

            # print("iteration no.",j)
            # print("Scl",Scl)

            ##Calculation for Smin and Smax
            size = np.shape(X)[0]
            arrayall = []
            # print(size)
            for m1 in range(size):
                centre1 = X[m1, :]
                points1 = (X[(m1 + 1):, :])
                distanceall = get_distances(centre1, points1)
                arrayall.append(distanceall)

            sorted_array = np.sort(np.concatenate(arrayall))
            element_number = np.shape(sorted_array)[0]
            Smin = np.sum(sorted_array[:N_value])
            Smax = np.sum(sorted_array[(element_number - N_value):])
            # print("Smin",Smin)
            # print("Smax",Smax)

            # calculation of C-index
            cindex = (Scl - Smin) / (Smax - Smin)
            Cindex_average.append(cindex)
            # print("C-index",cindex)

            # Update centroid location using the newly
            # assigned data point classes
            for c in range(z):
                centroids[c] = np.mean(X[classes == c], 0)

            # print("New Centroids",centroids)
            # print(Cindex_average)
        # print("Minimum C-index",np.amin(Cindex_average))
        # print("Average C-index",np.mean(Cindex_average))
        # print("K",z)
        minimum_cindex_values.append(np.amin(Cindex_average))
        average_cindex_values.append(np.mean(Cindex_average))
    print(minimum_cindex_values, "Min C-index")
    print(average_cindex_values, "Average C-index")
    print(k, "K values")

    #Plotting the Average and minimal C-index wrt to K

    group_colors = ['skyblue', 'coral', 'lightgreen', 'blue', 'brown', 'red', 'green', 'orange']

    fig1, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(k[:], minimum_cindex_values[:], linestyle='solid', color=group_colors, alpha=0.5)
    plt.suptitle('C-index Minimum Plot')
    ax.set_xlabel('$k$')
    ax.set_ylabel('$C-index min$')

    fig2, bx = plt.subplots(figsize=(4, 4))
    bx.scatter(k[:], average_cindex_values[:], linestyle='solid', color=group_colors, alpha=0.5)
    plt.suptitle('C-index Average Plot')
    bx.set_xlabel('$k$')
    bx.set_ylabel('$C-index avg$')
    plt.show()