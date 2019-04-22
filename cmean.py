import numpy as np
from sklearn.cluster import KMeans
import random as r
import matplotlib.pyplot as plt


def random_data_generator(dimension, limit=100, randomness=0.99):
    cell, data_temp = [], []

    for i in range(limit ** dimension):
        if r.random() > randomness:
            for d in range(dimension):
                cell.append(r.randrange(limit))
            data_temp.append(cell.copy())
            cell.clear()

    return data_temp


def random_membership_generator(data_in, dimension):
    membership = np.random.dirichlet(np.ones(len(data_in)))
    membership_temp = []

    for j in range(dimension):
        for i in range(len(data_in)):
            if j == 0:
                membership_temp.append([])
            membership_temp[i].append(membership[i])

    return membership_temp


def center_initialize(n_cluster, dimension, limit=100):
    cell, centers = [], []
    for n in range(n_cluster):
        for d in range(dimension):
            cell.append(r.randrange(limit))
        centers.append(cell.copy())
        cell.clear()

    return centers


def update_centers():
    pass


def update_membership():
    pass


def distance(x, c):
    dim = x.__len__()
    sum = 0
    for d in range(dim):
        sum += (x[d] - c[d]) ** 2

    return sum ** (0.5)



random_data = random_data_generator(2)
membership_matrix = random_membership_generator(random_data, 2)
cluster_centers = center_initialize(4, 2)
data = np.array(random_data)
centers = np.array(cluster_centers)


print('random data = ', random_data)
print('membership = ', membership_matrix)
print('centers = ', cluster_centers)










##################### TEST AREA #####################
# kmeans = KMeans(n_clusters=4)
# kmeans.fit(data)
# labels = kmeans.labels_
#
# # print(labels)
#
# plt.scatter(x=data[:, 0], y=data[:, 1], s=20, c=labels)
# plt.scatter(x=centers[:, 0], y=centers[:, 1], s=30, marker='D', c='black')
# plt.grid(True)
# plt.show()
#####################################################
