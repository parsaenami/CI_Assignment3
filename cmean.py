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


# TODO: change it like previous function
def random_membership_generator(data_in, dimension):
    membership = np.random.dirichlet(np.ones(len(data_in)))
    membership_temp = []

    for j in range(dimension):
        for i in range(len(data_in)):
            if j == 0:
                membership_temp.append([])
            membership_temp[i].append(membership[i])

    return membership_temp

# [
# [62, 5],
# [6, 1],
# [64, 11],
# [64, 44],
# [21, 64]
# ]

def center_initialize(n_cluster, dimension, limit=100):
    cell, centers = [], []
    for n in range(n_cluster):
        for d in range(dimension):
            cell.append(r.randrange(limit))
        centers.append(cell)

    return centers


random_data = random_data_generator(2)
membership_matrix = random_membership_generator(random_data, 2)
data = np.array(random_data)

print('random data = ', random_data)
print('membership = ', membership_matrix)

##################### TEST AREA #####################
kmeans = KMeans(n_clusters=4)
kmeans.fit(data)
labels = kmeans.labels_

# print(labels)

plt.scatter(x=data[:, 0], y=data[:, 1], s=30, c=labels)
plt.grid(True)
plt.show()
#####################################################
