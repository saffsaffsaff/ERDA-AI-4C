import numpy as np, pandas as pd
import matplotlib.pyplot as plt


class NN:
    def __init__(self, dataframe_path, no_types, no_carriers, no_neurons_first_layer, learning_rate,
                 cost_value_desired_batch):
        """
        Method to define Class parameters

        :param dataframe_path: path to DataFrame with processed data
        :param no_types: Number of different aircraft types
        :param no_carriers: Number of different fleet carriers
        :param no_neurons_first_layer: Number of neurons for the first layer
        :param cost_value_desired_batch:
        """
        # Import dataframe
        df = pd.read_pickle(dataframe_path)
        self.input_data = df[df.columns[4:]].to_numpy()
        self.no_inputs = self.input_data.shape[1]
        self.no_outputs = no_types + no_carriers
        self.correct_outputs = df[df.columns[2:5]].to_numpy()

        # Create weights matrices and bias vectors with random values
        # For the first layer
        self.weights_first_layer = np.random.rand(no_neurons_first_layer, self.no_inputs)
        self.bias_first_layer = np.random.rand(no_neurons_first_layer, 1)
        # For the second layer. This is the last layer before output layer, so it needs to have the same amount of
        # neurons
        no_neurons_second_layer = self.no_outputs
        self.weights_second_layer = np.random.rand(no_neurons_second_layer, no_neurons_first_layer)
        self.bias_second_layer = np.random.rand(no_neurons_second_layer, 1)
        # Assign other parameters
        self.learning_rate = learning_rate
        self.cost_value_desired = cost_value_desired_batch

    # Create activating functions and their derivatives
    @staticmethod
    def f_quadratic(x):
        return x**2

    @staticmethod
    def f_quadratic_derivative(x):
        return 2 * x

    @staticmethod
    def f_cubic(x):
        return x**3

    @staticmethod
    def f_cubic_derivative(x):
        return 3 * x**2

    @staticmethod
    def f_sigmoid(x):
        return 1/(1 + np.exp(-x))

    @staticmethod
    def f_sigmoid_derivative(x):
        return NN.f_sigmoid(x)(1 - NN.f_sigmoid(x))

    @staticmethod
    def f_cost(x, x_true):
        return (x - x_true) ** 2

    @staticmethod
    def f_cost_derivative(x, x_true):
        return 2 * (x - x_true)

    def nn_execution(self, input_data, expected_result):
        """

        Method to execute NN algorithm, return all the layers, and the cost function value

        :param input_data: An array of input values
        :param expected_result: An array of expected output_result
        :return:
        """
        first_linear_layer = self.weights_first_layer.dot(input_data) + self.bias_first_layer
        first_layer = NN.f_quadratic(first_linear_layer)
        second_linear_layer = self.weights_second_layer.dot(first_layer) + self.bias_second_layer
        second_layer = NN.f_cubic(second_linear_layer)
        output_layer = NN.f_sigmoid(second_layer)
        cost_vector = NN.f_cost(output_layer, expected_result)
        cost_total = np.sum(cost_vector)

        # Determine which aircraft is chosen

        return first_linear_layer, first_layer, second_linear_layer, second_layer, output_layer, cost_total

    def nn_gradient_function_calculation(self, input_data, first_layer, first_linear_layer, second_layer,
                                         second_linear_layer, output_layer, expected_result):
        """

        Method for gradient determination of Cost function with respect to weights and biases

        :param input_data: An array of input_data
        :param first_layer: An array with values of first layer of neurons
        :param first_linear_layer: An array with values of first layer of neurons before activating function was applied
        :param second_layer: An array with values of second layer of neurons
        :param second_linear_layer: An array with values of second layer of neurons before activating function was
        applied
        :param output_layer: An array with output layer of neurons
        :param expected_result: An array with expected/true result
        :return:
        """

        # Apply chain rule to get common part for all derivatives with respect to weights and biases at both layers
        dummy_derivative = NN.f_cost_derivative(output_layer, expected_result) * NN.f_sigmoid(second_layer) * \
                           NN.f_cubic_derivative(second_linear_layer)

        # Find derivatives with respect to all weights and biases.
        dc_db_first_layer = self.weights_second_layer.dot(NN.f_quadratic_derivative(first_linear_layer)) * \
                            dummy_derivative
        dc_dw_first_layer = (((np.tile(input_data, self.weights_first_layer.shape[0]).T *
                             NN.f_quadratic_derivative(first_linear_layer)).T * self.weights_second_layer.T).T *
                             dummy_derivative).T
        dc_dw_second_layer = (np.tile(first_layer, self.weights_second_layer.shape[0]).T * dummy_derivative).T
        dc_db_second_layer = dummy_derivative * 1

        return dc_dw_first_layer, dc_db_first_layer, dc_dw_second_layer, dc_db_second_layer

    def steepest_descent(self, batch_input_data, batch_expected_results, batch_size):
        """

        Function to apply steepest descent to update weight matrices and bias vectors for a single batch

        :param batch_input_data: The list containing input data for cases in batch
        :param batch_expected_results: The list containing expected results for cases in batch
        :param batch_size: The size of the batch
        :return:
        """
        # Setup the average batch cost to be higher than treshold, so that loop in initiated
        cost_batch = self.cost_value_desired + 1
        while cost_batch > self.cost_value_desired:
            # Setup lists to store the result from each case
            db_first_layer_batch = dw_first_layer_batch = dw_second_layer_batch = db_second_layer_batch = cost_batch = \
                []

            # Iterate over batch and append the results to storage lists
            for (input_data, expected_result) in (batch_input_data, batch_expected_results): # not sure whether this will work, may need zip function
                first_linear_layer, first_layer, second_linear_layer, second_layer, output_layer, cost_total = \
                    self.nn_execution(input_data, expected_result)
                dc_dw_first_layer, dc_db_first_layer, dc_dw_second_layer, dc_db_second_layer = \
                    self.nn_gradient_function_calculation(input_data, first_layer, first_linear_layer, second_layer,
                                                          second_linear_layer, output_layer, expected_result)
                db_first_layer_batch.append(- self.learning_rate * dc_db_first_layer)
                dw_first_layer_batch.append(- self.learning_rate * dc_dw_first_layer)
                db_second_layer_batch.append(- self.learning_rate * dc_db_second_layer)
                dw_second_layer_batch.append(- self.learning_rate * dc_dw_second_layer)
                cost_batch.append(cost_total)

            # Find average values for whole batch
            db_first_layer_batch = sum(db_first_layer_batch)/batch_size
            dw_first_layer_batch = sum(dw_first_layer_batch)/batch_size
            db_second_layer_batch = sum(db_second_layer_batch)/batch_size
            dw_second_layer_batch = sum(dw_second_layer_batch)/batch_size
            cost_batch = sum(cost_batch)/batch_size

            # Modify weight matrices and bias vectors
            self.bias_first_layer += db_first_layer_batch
            self.weights_first_layer += dw_first_layer_batch
            self.bias_second_layer += db_second_layer_batch
            self.weights_second_layer += dw_second_layer_batch
