import matplotlib.pyplot as plt
import numpy as np
from Dataset import Dataset
from PolynomialRegression import PolynomialRegression
from Metric import MSE
from MiniBatchGradientDescent import MiniBatchGradientDescent
from sklearn.preprocessing import StandardScaler

# Punto 2: Levantar el dataset enun arreglo (ds), graficar y partir en train %80, test %20
ds = Dataset('clase_8_dataset')
X = ds.data['entrada']
y = ds.data['salida']
plt.scatter(X, y)
plt.show()
X_norm = StandardScaler(with_std=True).fit_transform(X.reshape(-1, 1))
X_train, y_train, X_test, y_test = Dataset.split_dataset(X_norm, y, 0.8)

# Punto 3: Aplicar K-Folds con regresión polinómica y elegir orden del polinomio para mejor aproximacion
orden = np.array([1, 2, 3, 4])
kfolds_lr = np.zeros(orden.shape)
for g in range(orden.shape[0]):
    # Definimos el modelo a emplear
    p_regression = PolynomialRegression(g+1)
    # Definimos una métrica
    error = MSE()
    chunk_size = int(len(X_train) / 5)
    mse_list = []
    for i in range(0, len(X_train), chunk_size):
        end = i + chunk_size if i + chunk_size <= len(X_train) else len(X_train)
        new_X_valid = X_train[i: end]
        new_y_valid = y_train[i: end]
        new_X_train = np.concatenate([X_train[: i], X_train[end:]])
        new_y_train = np.concatenate([y_train[: i], y_train[end:]])

        p_regression.fit(new_X_train, new_y_train.reshape(-1, 1))
        prediction = p_regression.predict(new_X_valid)
        k_mse = error(new_y_valid.reshape(-1, 1), prediction)
        mse_list.append(k_mse)

    kfolds_lr[g] = np.mean(mse_list)

plt.plot(orden, kfolds_lr)
plt.show()
# Una vez aplicad KFolds podemos visualizar que los polinomios de grado 3 y 4 ya poseen los
# menors errores, y por simplificación de algoritmos, se elige el grado 3 para continuar.

# Grafica del polinomio elegido:
p_regression = PolynomialRegression(3)
yp_train = p_regression.fit_transform(X_train, y_train)
yp_test = p_regression.predict(X_test)

x_cl = p_regression.model
x = np.linspace(X_test.min(), X_test.max(), 100)
y_cl = x_cl[0] * x**0 + x_cl[1] * x**1 + x_cl[2] * x**2 + x_cl[3] * x**3

plt.plot(x,y_cl)
plt.scatter(X_test, y_test)
plt.show()

# Punto 4: Ahora aplicamos MiniBatch Gradiente Descendiente
MB = MiniBatchGradientDescent(alpha=0.1, n_epochs=50, n_batches=15, poly=3, lbd=0.001)
MB.fit(X_train, y_train)

y_pred = MB.predict(X_test)
plt.scatter(X_test, y_test)
plt.scatter(X_test, y_pred, color='red')
plt.show()

# Camparar modelos de MiniBatch y polinomico
x_mb = MB.model
x = np.linspace(X_test.min(), X_test.max(), 100)
y_mb = x_mb[0] * x**0 + x_mb[1] * x**1 + x_mb[2] * x**2 + x_mb[3] * x**3

print(x_cl.T)
print(x_mb)
plt.plot(x,y_cl, color='red')
plt.plot(x,y_mb)
plt.show()