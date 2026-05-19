from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)

    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] -= self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu):
        super().__init__(init_lr, model)
        self.mu = mu
        self.velocity = {}
        for layer_idx, layer in enumerate(self.model.layers):
            if layer.optimizable:
                for key, value in layer.params.items():
                    self.velocity[(layer_idx, key)] = np.zeros_like(value)

    def step(self):
        for layer_idx, layer in enumerate(self.model.layers):
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    velocity_key = (layer_idx, key)
                    self.velocity[velocity_key] = self.mu * self.velocity[velocity_key] - self.init_lr * layer.grads[key]
                    layer.params[key] += self.velocity[velocity_key]
