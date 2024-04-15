"""
A loss function measures how well a model is doing. It takes the model's predictions and the correct labels and returns a single number (a scalar) representing how well the model did.
"""
import numpy as np

from vsdl.tensor import Tensor

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError
    
class MSE(Loss):
    """
    MSE is mean squared error, although we actually return the sum of squared errors.
    """
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)