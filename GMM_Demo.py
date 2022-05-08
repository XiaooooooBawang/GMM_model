# Author: XiaooooooBawang <1206286599@qq.com>
# License: BSD-3-Claus

import matplotlib.pyplot as plt
import numpy as np
from model._gmm import GMM

data = np.loadtxt('gmm.data')
train_data = np.array(data)

# minmax scale data
for i in range(data.shape[1]):
    max_ = data[:, i].max()
    min_ = data[:, i].min()
    train_data[:, i] = (data[:, i] - min_) / (max_ - min_)


# create the model and deploy it on gpu or cpu
model = GMM(2, 'cpu')  # input the number of component in GMM
model.initial(train_data)  # use the shape of train_data to initialize the params of model
cluster, train_params = model.train(100, train_data)
class1 = np.array([data[i] for i in range(data.shape[0]) if cluster[i] == 0])
class2 = np.array([data[i] for i in range(data.shape[0]) if cluster[i] == 1])

# print("Phi:")
# print(train_params.Phi)
# print("Mu:")
# print(train_params.Mu)
# print("Sigma:")
# print(train_params.Sigma)

plt.plot(class1[:, 0], class1[:, 1], 'rs', label="class1")
plt.plot(class2[:, 0], class2[:, 1], 'bo', label="class2")
plt.legend(loc="best")
plt.show()

# save the model after training
model.save()
