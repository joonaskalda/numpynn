from nn import TwoLayerNet

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os # processing file path
import gzip # unzip the .gz file, not used here
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print("Here are the input datasets: ")
#print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.

train_df = pd.read_csv('input/fashion-mnist_train.csv',sep=',')
test_df = pd.read_csv('input/fashion-mnist_test.csv', sep = ',')

train_data = np.array(train_df, dtype = 'float32')
test_data = np.array(test_df, dtype = 'float32')

X_train = train_data[:,1:]

y_train = train_data[:,0].astype(int)

X_test= test_data[:,1:]

y_test=test_data[:,0].astype(int)

m_train = X_train.shape[0]
m_test = X_test.shape[0]

# random check with nine training examples
# np.random.seed(0);
# indices = list(np.random.randint(m_train,size=9))
# for i in range(9):
#     plt.subplot(3,3,i+1)
#     plt.imshow(X_train[indices[i]].reshape(28,28), cmap='gray', interpolation='none')
#     plt.title("Index {} Class {}".format(indices[i], y_train[indices[i]]))
#     plt.tight_layout()
# plt.show()
# Subsample the data
m_train = 59000
m_validation = 1000

mask = list(range(m_train, m_train + m_validation))
X_val = X_train[mask]
y_val = y_train[mask]

mask = list(range(m_train))
X_train = X_train[mask]
y_train = y_train[mask]

mask = list(range(m_test))
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape data to rows
X_train = X_train.reshape(m_train, -1)
X_val = X_val.reshape(m_validation, -1)
X_test = X_test.reshape(m_test, -1)

input_size = X_train.shape[1]
hidden_size = 10
num_classes = 10
net = TwoLayerNet(input_size, hidden_size, num_classes)

# Train the network
stats = net.train(X_train, y_train, X_val, y_val,
            num_epochs=10, batch_size=1024,
            learning_rate=7.5e-4,
            reg=1.0, verbose=True)  


# Predict on the validation set
val_acc = (net.predict(X_val) == y_val).mean()


plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
plt.subplots_adjust(wspace =0, hspace =0.3)


# Plot the loss function and train / validation accuracies

plt.subplot(2, 1, 1)
plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(stats['train_acc_history'], label='train')
plt.plot(stats['val_acc_history'], label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.legend()
plt.show()