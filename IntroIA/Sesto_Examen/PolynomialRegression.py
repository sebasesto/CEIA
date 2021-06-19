import numpy as np
from sklearn.preprocessing import PolynomialFeatures

class LinearRegression():
    def fit(self, x, y):
        if x.ndim == 1:
            w = (x.T @ y) / (x.T @ x)
        else:
            w = np.linalg.inv(x.T @ x) @ x.T @ y
        self.model = w

    def predict(self, x):
        return self.model @ x

class PolynomialRegression(LinearRegression):
    def __init__(self, grade):
        self.poly = PolynomialFeatures(grade)

    def fit(self, x, y):
        res = self.poly.fit_transform(x.reshape(-1, 1))
        super().fit(res, y.reshape(-1,1))

    def predict(self, x):
        return self.poly.fit_transform(x.reshape(-1, 1)) @ self.model

    def fit_transform(self, x, y):
        self.fit(x,y)
        return self.predict(x).reshape(1,-1)

# from BaseModel import BaseModel
# import numpy as np
# from sklearn.preprocessing import PolynomialFeatures
# from LinearRegression import LinearRegression
#
#
# class PolynomialRegression(LinearRegression):
#
#     def __init__(self, grade):
#         self.poly = PolynomialFeatures(grade)
#
#     def fit(self, x, y):
#         if x.ndim != 1:
#             print("Input must have only one feature. Multi-feature is not yet implemented.1")
#             return
#         res = self.poly.fit_transform(x.reshape(-1, 1))
#         super().fit(res, y.reshape(-1, 1))
#         #super().fit(res, y)
#
#     def predict(self, x):
#         if x.ndim != 1:
#             print("Input must have only one feature. Multi-feature is not yet implemented.2")
#             return
#         return self.poly.fit_transform(x.reshape(-1, 1)) @ self.model
#
#
#     def fit_transform(self, x, y):
#         self.fit(x,y)
#         return self.predict(x).reshape(1,-1)
