import numpy as np
from sklearn.cluster import KMeans
import random as r
import matplotlib.pyplot as plt

def random_data_generator():
    data_temp = []

    for i in range(100):
        for j in range(100):
            if r.random() > 0.99:
                data_temp.append([r.randrange(100), r.randrange(100)])

    # random_membership_generator(data_temp)

    return data_temp


def random_membership_generator(data_in):
    membership1 = np.random.dirichlet(np.ones(len(data_in)))
    membership2 = np.random.dirichlet(np.ones(len(data_in)))
    membership_temp = []

    for i in range(len(data_in)):
        membership_temp.append([membership1[i], membership2[i]])

    return membership_temp


random_data = random_data_generator()
membership_matrix = random_membership_generator(random_data)
data = np.array(random_data)

##################### TEST AREA #####################

kmeans = KMeans(n_clusters=4)
kmeans.fit(data)
labels = kmeans.labels_

# print(labels)

plt.scatter(x=data[:,0], y=data[:,1], s=30, c=labels)
plt.grid(True)
plt.show()

#####################################################