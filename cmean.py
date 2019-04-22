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

        for i in range((self.limit ** self.dimension)):
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
            for cc in range(len(self.C[c])):
                for u in range(len(self.U)):
                    sum0 += self.U[u][c] ** self.m
                    sum1 += ((self.U[u][c] ** self.m) * self.X[u][cc])
                self.C[c][cc] = sum1 / sum0
                sum0, sum1 = 0, 0

    def update_membership(self):
        sum0 = 0

        for i in range(len(self.U)):
            for j in range(len(self.U[i])):
                for k in range(len(self.C)):
                    sum0 += (distance(self.X[i], self.C[j]) / distance(self.X[i], self.C[k])) ** (2 / (self.m - 1))
                self.U[i][j] = sum0 ** (-1)
                sum0 = 0

    def coloring(self):
        colors_out = []

        for x in range(len(self.X)):
            colors_out.append(self.U[x].index(max(self.U[x])))

        return colors_out


if __name__ == '__main__':
    c_mean = FuzzyCMean(2, 3, 50, randomness=0.96, limit=100)

    for it in range(c_mean.iterations):
        c_mean.update_centers()
        c_mean.update_membership()

    data = np.array(c_mean.X)
    centers = np.array(c_mean.C)
    colors = np.array(c_mean.coloring())

    plt.scatter(x=data[:, 0], y=data[:, 1], s=10, c=colors)
    plt.scatter(x=centers[:, 0], y=centers[:, 1], s=30, marker='D', c='red', label='Centers')
    plt.legend()
    plt.minorticks_on()
    plt.grid(True)
    plt.title(f'Fuzzy C-Means Clustering:\n{c_mean.c_cluster} clusters / {c_mean.iterations} iterations / {c_mean.X.__len__()} data in {c_mean.dimension} dimensions')
    plt.savefig('FuzzyCMeansClustering.png', bbox_inches='tight')
    plt.show()
