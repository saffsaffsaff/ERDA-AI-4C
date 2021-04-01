import numpy as np
import matplotlib.pyplot as plt


class NN:
    def __init__(self, data, no_input_values, no_output_values, no_neurons_first_layer):
        """
        Method to define Class parameters

        :param data: DataFrame with processed data
        :param no_input_values: Number of input variables - needs change to automatic detection
        :param no_output_values: Number of output variables - needs change to automatic detection
        :param no_neurons_first_layer: Number of neurons for the first layer
        :param no_neurons_second_layer: Number of neurons for the second layer
        """
        # Import dataframe
        # Create wights matrices and bias vectors with random values
        # For the first layer
        self.weights_first_layer = np.random.rand(no_input_values, no_input_values)
        self.bias_first_layer = np.random.rand(no_neurons_first_layer, 1)
        # For the second layer. This is the last layer before output layer, so it needs to have the same amount of
        # neurons
        no_neurons_second_layer = no_output_values
        self.weights_second_layer = np.random.rand(no_neurons_first_layer, no_neurons_first_layer)
        self.bias_second_layer = np.random.rand(no_neurons_second_layer, 1)

    @staticmethod
    def f_quadratic(x):
        y = x**2
        return y

    @staticmethod
    def f_quadratic_derivative(x):
        y = 2 * x
        return y

    @staticmethod
    def f_cubic(x):
        y = x**3
        return y

    @staticmethod
    def f_cubic_derivative(x):
        y = 3 * x**2
        return y

    @staticmethod
    def f_sigmoid(x):
        y = 1/(1 + np.exp(-x))
        return y

    @staticmethod
    def f_sigmoid_derivative(x):
        y = NN.f_sigmoid(x)(1 - NN.f_sigmoid(x))
        return y

    @staticmethod
    def f_cost(x, x_true):
        y = (x - x_true) ** 2
        return y

    @staticmethod
    def f_cost_derivative(x, x_true):
        y = 2 * (x - x_true)
        return y

    def nn_execution(self, input_data, expected_result):
        """

        Method to execute NN algorithm and return all the layers

        :param input_data: An array of input values
        :param expected_result: An array of expected output_result
        :return:
        """

        first_layer = NN.f_quadratic(self.weights_first_layer * input_data + self.bias_first_layer)
        second_layer = NN.f_cubic(self.weights_second_layer * input_data + self.bias_second_layer)
        output_layer = NN.f_sigmoid(second_layer)
        cost_vector = NN.f_cost(output_layer, expected_result)
        cost_total = np.sum(cost_vector)

        # Determine which aircraft is chosen

        return first_layer, second_layer, output_layer, cost_total

    def nn_gradient_function_calculation(self, input_data, first_layer, second_layer, output_layer, expected_result):
        """

        Method for gradient determination of Cost function with respect to weights and biases

        :param input_data: An array of input_data
        :param first_layer: An array with values of first layer of neurons
        :param second_layer: An array with values of second layer of neurons
        :param output_layer: An array with output layer of neurons
        :param expected_result: An array with expected/true result
        :return:
        """
        # Apply chain rule to get common part for all derivatives with respect to wieghts and biases at both layers
        dummy_derivative = NN.f_cost_derivative(output_layer, expected_result) * NN.f_sigmoid(second_layer) * \
                             NN.f_cubic_derivative(self.weights_second_layer * first_layer + self.bias_second_layer)

        # Find derivatives with respect to all weights and biases
        dc_dw_first_layer =
        dc_db_first_layer =
        dc_dw_second_layer = np.tile(first_layer,)
        dc_db_second_layer = dummy_derivative * 1
