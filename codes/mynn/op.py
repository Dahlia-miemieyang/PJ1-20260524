from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        scale = np.sqrt(2.0 / in_dim)
        self.W = initialize_method(size=(in_dim, out_dim)) * scale
        self.b = np.zeros((1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return X @ self.W + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads['W'] = self.input.T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.W.T
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size)) * scale
        self.b = np.zeros((1, out_channels, 1, 1))
        self.grads = {'W' : None, 'b' : None}
        self.params = {'W' : self.W, 'b' : self.b}
        self.input = None
        self.padded_input = None
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [out, in, k, k]
        """
        self.input = X
        if self.padding > 0:
            self.padded_input = np.pad(
                X,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            )
        else:
            self.padded_input = X

        batch_size, _, height, width = self.padded_input.shape
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(out_height):
                    h_start = i * self.stride
                    h_end = h_start + self.kernel_size
                    for j in range(out_width):
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size
                        region = self.padded_input[b, :, h_start:h_end, w_start:w_end]
                        output[b, oc, i, j] = np.sum(region * self.W[oc]) + self.b[0, oc, 0, 0]
        return output

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        batch_size, _, out_height, out_width = grads.shape
        grad_input_padded = np.zeros_like(self.padded_input)
        self.grads['W'] = np.zeros_like(self.W)
        self.grads['b'] = np.sum(grads, axis=(0, 2, 3), keepdims=True).reshape(1, self.out_channels, 1, 1)

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(out_height):
                    h_start = i * self.stride
                    h_end = h_start + self.kernel_size
                    for j in range(out_width):
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size
                        grad_value = grads[b, oc, i, j]
                        region = self.padded_input[b, :, h_start:h_end, w_start:w_end]
                        self.grads['W'][oc] += grad_value * region
                        grad_input_padded[b, :, h_start:h_end, w_start:w_end] += grad_value * self.W[oc]

        if self.padding > 0:
            return grad_input_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return grad_input_padded
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.labels = None
        self.probs = None
        self.predicts = None
        self.grads = None
        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.predicts = predicts
        self.labels = labels.astype(np.int64)

        if self.has_softmax:
            self.probs = softmax(predicts)
        else:
            self.probs = predicts

        clipped = np.clip(self.probs, 1e-12, 1.0)
        losses = -np.log(clipped[np.arange(self.labels.shape[0]), self.labels])
        return np.mean(losses)

    def backward(self):
        # first compute the grads from the loss to the input
        batch_size = self.labels.shape[0]
        grads = self.probs.copy()
        grads[np.arange(batch_size), self.labels] -= 1
        self.grads = grads / batch_size
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition