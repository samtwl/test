import numpy as np

np.random.seed(5)


class CrossEntropyLoss:
    """
    https://numpy-ml.readthedocs.io/en/latest/numpy_ml.neural_nets.losses.html#crossentropy
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes

    def loss(self, y_true, pred):
        """_summary_

        Args:
            y_true (_type_): A single value
            pred (_type_): _description_

        Returns:
            _type_: _description_
        """
        true_one_hot = oneHotEncoding.encode([y_true], self.n_classes)
        self.true = true_one_hot
        pred = np.clip(pred, 1e-7, 1)  # To prevent np.log(0)
        self.pred = pred
        pred = np.log(pred)
        loss_for_each = -np.dot(true_one_hot, pred)

        return loss_for_each

    def batch_loss(self, y_true, pred):
        """_summary_

        Args:
            y_true (_type_): A single array of the true class, e.g. [1, 0, 3]
            pred (_type_): _description_

        Returns:
            _type_: _description_
        """
        true_one_hot = oneHotEncoding.encode(y_true, self.n_classes)
        pred = np.clip(pred, 1e-7, 1)  # To prevent np.log(0)
        pred = np.log(pred)
        loss_for_each = np.dot(true_one_hot, pred.T)
        total_loss = -np.sum(loss_for_each, axis=1)[0]
        return total_loss

    def backward(self):
        return self.pred - self.true.T


class MLPTwoLayers:

    # DO NOT adjust the constructor params
    def __init__(
        self,
        input_size=3072,
        hidden_size=100,
        output_size=10,
        loss_func=CrossEntropyLoss,
    ):
        self.input_size = input_size
        self.hidden_size = (hidden_size,)
        self.output_size = output_size
        self.learning_rate = 1e-3
        self.activations = ReLU()
        self.input_layer = Layer("input_layer", input_size, hidden_size)
        self.hidden_layers = [
            Layer("hidden_1", hidden_size, 10),
            Layer("hidden_2", 10, 5),
        ]
        self.output_layer = Layer("output_layer", 5, output_size)
        self.final_activation = Softmax()
        self.loss_func = loss_func(output_size)

    def forward(self, features):
        """
        Takes in the features
        returns the prediction
        """
        reshaped = features.reshape(-1, 1)
        output = self.input_layer.forward(reshaped)
        output = self.activations.forward(output)
        for layer in self.hidden_layers:
            output = layer.forward(output)
            output = self.activations.forward(output)
        output = self.output_layer.forward(output)
        output = self.final_activation.forward(output)
        return output

    def loss(self, predictions, label):
        """
        Takes in the predictions and label
        returns the training loss
        """
        loss_val = self.loss_func.loss(label, predictions)
        return loss_val

    def backward(self):
        """
        Adjusts the internal weights/biases

        Args:
            X: the input array
            y: the true label. Single layer e.g. [1,0,2,1]
            output: The predicted probability of each classes from the model. Shape of (num_output, num_classes)
                e.g. [[0.3, 0.1, 0.9]]
        """
        d_loss = self.loss_func.backward()
        # d_softmax = self.final_activation.backward(d_loss)
        d_output = self.output_layer.backward(d_loss, self.learning_rate)
        for layer in self.hidden_layers[::-1]:
            d_output = self.activations.backward(d_output)
            d_output = layer.backward(d_output, self.learning_rate)
        d_output = self.activations.backward(d_output)
        self.input_layer.backward(d_output, self.learning_rate)
        return d_loss

    def get_weight(self):
        weights = {}
        biases = {}
        weights["output"] = self.output_layer.weights
        biases["output"] = self.output_layer.biases
        weights["input"] = self.input_layer.weights
        biases["input"] = self.input_layer.biases
        for i, layer in enumerate(self.hidden_layers):
            weights["hidden_" + str(i)] = layer.weights
            biases["biases_" + str(i)] = layer.biases
        return weights, biases


class ReLU:
    """No, ReLU has derivative.
    ReLU function f(x)=max(0,x).
     It means if x<=0 then f(x)=0, else f(x)=x.

     Derivatives:
        when x<0 so the derivative of f(x) with respect to x gives result f'(x)=0.
        else: f'(x)=1.
    """

    def __init__(self) -> None:
        pass

    def forward(self, input):
        relu_forward = np.maximum(0, input)
        return relu_forward

    def backward(self, input):
        """Compute the graident of loss w.r.t ReLU inputs

        Args:
            input (_type_): _description_
            grad_output (_type_): _description_
        """
        relu_grad = (input > 0) * 1
        return relu_grad


class Softmax:
    def __init__(self) -> None:
        pass

    def forward(self, scores):
        self.output = np.exp(scores) / np.sum(np.exp(scores))
        return self.output

    def backward(self, output_gradient):
        """The derivative of softmax

        if i == j; Pi * (1 - Pi)
        else: -Pi * Pj

        Args:
            input (_type_): _description_

        Returns:
            _type_: _description_
        """
        n = np.size(self.output)
        top = (np.identity(n) - self.output.T) * self.output
        return np.dot(top, output_gradient.T)


class Layer:
    def __init__(
        self,
        name: str,
        n_inputs: int,
        n_outputs: int,
        biases: np.ndarray = None,
        weights: np.ndarray = None,
    ):
        self.name = name
        self.weights = (
            weights if weights is not None else np.random.randn(n_outputs, n_inputs)
        )
        self.biases = biases if biases is not None else np.random.randn(n_outputs, 1)
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.weights, self.inputs) + self.biases
        return self.output

    def backward(self, output_gradient, learning_rate):
        # print(output_gradient, self.inputs.T)
        weights_gradient = np.dot(
            output_gradient, self.inputs.T
        )  # Derivate of the error, with respect to the weight
        # print("Updating..", weights_gradient, self.name, learning_rate)
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * output_gradient
        return np.dot(self.weights.T, output_gradient)


class oneHotEncoding:
    def __init__(self) -> None:
        pass

    @staticmethod
    def encode(value: np.ndarray, num_classes: int):
        # Initialsing empty array of (len(value), num_classes)
        encoded_array = np.zeros((len(value), num_classes), dtype=int)
        # Labeling respective index with value 1
        encoded_array[np.arange(len(value)), value] = 1
        return encoded_array
