import numpy as np

class MSE():
    def __call__(self, target, prediction):
        # n = target.size
        return np.mean((target - prediction) ** 2)