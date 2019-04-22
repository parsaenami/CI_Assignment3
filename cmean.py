import numpy as np
from sklearn.cluster import KMeans
import random as r
import matplotlib.pyplot as plt


def distance(x, c):
    dim = x.__len__()
    sum0 = 0

    for d in range(dim):
        sum0 += (x[d] - c[d]) ** 2

    return sum0 ** 0.5


class FuzzyCMean:

    def __init__(self, dimension, c_cluster, iterations, limit=100, randomness=0.99, m=2):
        self.iterations = iterations
        self.m = m
        self.c_cluster = c_cluster
        self.randomness = randomness
        self.limit = limit
        self.dimension = dimension

        self.C = self.center_initialize()
        self.X = self.random_data_generator()
        self.U = self.random_membership_generator()

    def random_data_generator(self):
        cell, data_temp = [], []

        for i in range(self.limit ** self.dimension):
            if r.random() > self.randomness:
                for d in range(self.dimension):
                    cell.append(r.randrange(self.limit))

                data_temp.append(cell.copy())
                cell.clear()

        return data_temp

    def random_membership_generator(self):
        membership_temp, membership_list = [], []

        for n in range(self.c_cluster):
            membership_list.append(np.random.dirichlet(np.ones(len(self.X))))

        for j in range(self.c_cluster):
            for i in range(len(self.X)):
                if j == 0:
                    membership_temp.append([])

                membership_temp[i].append(membership_list[j][i])

        return membership_temp

    def center_initialize(self):
        cell, centers_out = [], []

        for n in range(self.c_cluster):
            for d in range(self.dimension):
                cell.append(r.randrange(self.limit))

            centers_out.append(cell.copy())
            cell.clear()

        return centers_out

    def update_centers(self):
        sum0, sum1 = 0, 0

        for c in range(len(self.C)):
            for u in self.U:
                for x in self.X:
                    sum0 += u[c] ** self.m
                    sum1 += (u[c] ** self.m) * x[c]

            self.C[c] = sum1 / sum0
            sum0, sum1 = 0, 0

    def update_membership(self):
        sum0 = 0

        for i in range(len(self.U)):
            for j in range(len(self.U[i])):
                for k in range(len(self.C)):
                    sum0 += (distance(self.X[i], self.C[j]) / distance(self.X[i], self.C[k])) ** (2 / (self.m - 1))
                self.U[i][j] = sum0 ** (-1)
                sum0 = 0


c_mean = FuzzyCMean(2, 4, 100)
# random_data = c_mean.random_data_generator()
# membership_matrix = c_mean.random_membership_generator(random_data)
# cluster_centers = c_mean.center_initialize()
data = np.array(c_mean.X)
centers = np.array(c_mean.C)

# print('random data = ', c_mean.X)
# print('membership = ', c_mean.U)
# print('centers = ', c_mean.C)

# s0, s1, s2, s3 = 0, 0, 0, 0
# for y in c_mean.U:
#     s0 += y[0]
#     s1 += y[1]
#     s2 += y[2]
#     s3 += y[3]
# print('s0 = ', s0)
# print('s1 = ', s1)
# print('s2 = ', s2)
# print('s3 = ', s3)

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
