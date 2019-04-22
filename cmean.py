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

    random_membership_generator(data_temp)

    return data_temp


def random_membership_generator(data_in):
    membership1 = np.random.dirichlet(np.ones(len(data_in)))
    membership2 = np.random.dirichlet(np.ones(len(data_in)))
    membership_temp = []

    for i in range(len(data_in)):
        membership_temp.append([membership1[i], membership2[i]])

    return membership_temp