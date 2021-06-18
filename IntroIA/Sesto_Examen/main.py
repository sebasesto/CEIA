from Dataset import Dataset
import matplotlib.pyplot as plt
from PolynomialRegression import PolynomialRegression
from Metric import Accuracy
import numpy as np

# Import dataset and split in training and test sets
ds = Dataset('clase_8_dataset')
training_x, training_y, test_x, test_y = Dataset.split_dataset(ds.data['Entrada'], ds.data['Salida'], 0.8)
plt.scatter(training_x, training_y)
plt.show()

def k_folds(X_train, y_train, grado, k=5):
    # Definimos el modelo a emplear
    p_regression = PolynomialRegression(grado)

    # Definimos una métrica
    error = Accuracy()

    chunk_size = int(len(X_train) / k)
    acc_list = []

    for i in range(0, len(X_train), chunk_size):
        end = i + chunk_size if i + chunk_size <= len(X_train) else len(X_train)
        new_X_valid = X_train[i: end]
        new_y_valid = y_train[i: end]
        new_X_train = np.concatenate([X_train[: i], X_train[end:]])
        new_y_train = np.concatenate([y_train[: i], y_train[end:]])

        p_regression.fit(new_X_train, new_y_train.reshape(-1, 1))
        prediction = p_regression.predict(new_X_valid)
        k_error = error(new_y_valid, prediction)
        acc_list.append(k_error)

    mean_acc = np.mean(acc_list)
    return mean_acc

# K-Folds para mejor aproximacion
lr_list = np.array([1, 2, 3, 4])
kfolds_lr = np.zeros(lr_list.shape)
for i in range (lr_list.shape[0]):
    kfolds_lr[i] = k_folds(training_x, training_y.reshape(-1, 1), i+1)

mejor_lr = lr_list[np.argmax(kfolds_lr)]
print(mejor_lr)
