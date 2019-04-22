import numpy as np
from sklearn.cluster import KMeans
import random as r
import matplotlib.pyplot as plt


class FuzzyCMean:

    def __init__(self, dimension, c_cluster, limit=100, randomness=0.99):
        self.c_cluster = c_cluster
        self.randomness = randomness
        self.limit = limit
        self.dimension = dimension

    def random_data_generator(self):
        cell, data_temp = [], []

        for i in range(self.limit ** self.dimension):
            if r.random() > self.randomness:
                for d in range(self.dimension):
                    cell.append(r.randrange(self.limit))
                data_temp.append(cell.copy())
                cell.clear()

        return data_temp

    def random_membership_generator(self, data_in):
        membership = np.random.dirichlet(np.ones(len(data_in)))
        membership_temp = []

        for j in range(self.dimension):
            for i in range(len(data_in)):
                if j == 0:
                    membership_temp.append([])
                membership_temp[i].append(membership[i])

        return membership_temp

    def center_initialize(self):
        cell, centers = [], []
        for n in range(self.c_cluster):
            for d in range(self.dimension):
                cell.append(r.randrange(self.limit))
            centers.append(cell.copy())
            cell.clear()

        return centers

    def update_centers(self, ):
        pass

    def update_membership(self, ):
        pass

    def distance(self, x, c):
        dim = x.__len__()
        sum = 0
        for d in range(dim):
            sum += (x[d] - c[d]) ** 2

        return sum ** (0.5)


cmean = FuzzyCMean(2, 4)
random_data = cmean.random_data_generator()
membership_matrix = cmean.random_membership_generator(random_data)
cluster_centers = cmean.center_initialize()
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
