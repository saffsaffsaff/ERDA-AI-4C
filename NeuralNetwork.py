import numpy as np, pandas as pd, random
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
        self.no_inputs, self.no_input_neurons = self.input_data.shape
        self.no_outputs = no_types + no_carriers
        self.correct_outputs = df[df.columns[2:4]].to_numpy()
        # create numpy array of the correct output values for each neuron (number of images)X(number of output neurons)
        self.correct_outputs_nn_format = np.zeros((len(self.correct_outputs), self.no_outputs))
        for i, img in enumerate(self.correct_outputs):
            self.correct_outputs_nn_format[i][int(img[0])] = 1
            self.correct_outputs_nn_format[i][int(img[1]) + no_types] = 1

        # Create weights matrices and bias vectors with random values
        # For the first layer
        self.weights_first_layer = np.random.rand(no_neurons_first_layer, self.no_input_neurons)
        self.bias_first_layer = np.random.rand(no_neurons_first_layer)
        # For the second layer. This is the last layer before output layer, so it needs to have the same amount of
        # neurons
        no_neurons_second_layer = self.no_outputs
        self.weights_second_layer = np.random.rand(no_neurons_second_layer, no_neurons_first_layer)
        self.bias_second_layer = np.random.rand(no_neurons_second_layer)
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
        dc_dw_first_layer = (self.weights_second_layer.dot((np.tile(input_data, (self.weights_first_layer.shape[0], 1)).T *
                             NN.f_quadratic_derivative(first_linear_layer)).T).T * dummy_derivative).T
        dc_dw_second_layer = (np.tile(first_layer, (self.weights_second_layer.shape[0], 1)).T * dummy_derivative).T
        dc_db_second_layer = dummy_derivative * 1

        return dc_dw_first_layer, dc_db_first_layer, dc_dw_second_layer, dc_db_second_layer

    def update_weights_and_biases(self, batch_input_data, batch_expected_results):
        """
        Function to apply steepest descent to update weight matrices and bias vectors for a single batch

        :param batch_input_data: The list containing input data for cases in batch
        :param batch_expected_results: The list containing expected results for cases in batch
        :return:
        """
        batch_size = batch_input_data.shape[0]
        # Setup the average batch cost to be higher than threshold, so that loop is initiated
        batch_average_cost = self.cost_value_desired + 1
        iteration = 0
        # Setup the lists to store iteration number and average cost batch on that iteration:
        batch_average_costs_storage, iterations_storage = [], []
        # Initiate loop
        while batch_average_cost > self.cost_value_desired:
            # Setup lists to store the results from each case
            db_first_layer_storage, dw_first_layer_storage, dw_second_layer_storage, db_second_layer_storage,\
            cases_costs_storage = [], [], [], [], []

            # Iterate over batch and to get the results of each case
            for (input_data, expected_result) in zip(batch_input_data, batch_expected_results):
                first_linear_layer, first_layer, second_linear_layer, second_layer, output_layer, case_cost = \
                    self.nn_execution(input_data, expected_result)
                dc_dw_first_layer, dc_db_first_layer, dc_dw_second_layer, dc_db_second_layer = \
                    self.nn_gradient_function_calculation(input_data, first_layer, first_linear_layer, second_layer,
                                                          second_linear_layer, output_layer, expected_result)

                # Append the case results of storage lists

                db_first_layer_storage.append(- self.learning_rate * dc_db_first_layer)
                dw_first_layer_storage.append(- self.learning_rate * dc_dw_first_layer)
                db_second_layer_storage.append(- self.learning_rate * dc_db_second_layer)
                dw_second_layer_storage.append(- self.learning_rate * dc_dw_second_layer)
                cases_costs_storage.append(case_cost)

            # Find average values for whole batch
            db_first_layer_average = sum(db_first_layer_storage) / batch_size
            dw_first_layer_average = sum(dw_first_layer_storage) / batch_size
            db_second_layer_average = sum(db_second_layer_storage) / batch_size
            dw_second_layer_average = sum(dw_second_layer_storage) / batch_size
            batch_average_cost = sum(cases_costs_storage) / batch_size

            # Update the iteration number and store the average cost value of the batch
            iteration += 1
            iterations_storage.append(iteration)
            batch_average_costs_storage.append(batch_average_cost)

            # Modify weight matrices and bias vectors
            self.bias_first_layer += db_first_layer_average
            self.weights_first_layer += dw_first_layer_average
            self.bias_second_layer += db_second_layer_average
            self.weights_second_layer += dw_second_layer_average

        fig = plt.figure()
        convergence_graph = fig.add_subplot(111, title="Average cost value per iteration")
        convergence_graph.plot(iterations_storage, batch_average_costs_storage)
        convergence_graph.x_label("Batch average cost value")
        convergence_graph.y_label("Iteration")
        plt.show()

    def check_accuracy(self, training_data, training_data_output, checking_data, checking_data_output):
        cost_training = [self.nn_execution(input_data, output_data)[-1] for input_data, output_data in
                         zip(training_data, training_data_output)]
        cost_checking = [self.nn_execution(input_data, output_data)[-1] for input_data, output_data in
                         zip(checking_data, checking_data_output)]

        print((min(cost_training), max(cost_training), sum(cost_training) / len(cost_training)),
              (min(cost_checking), max(cost_checking), sum(cost_checking) / len(cost_checking)))

    def train(self, batch_size: int, training_data_fraction: float):
        # split input data into training data and checking data based on the training data fraction
        print("splitting data ...  ", end='')
        random_range = random.sample(range(self.no_inputs), self.no_inputs)
        training_data, training_data_output = zip(*[(self.input_data[i], self.correct_outputs_nn_format[i]) for i in
                                                    random_range[:int(self.no_inputs * training_data_fraction)]])
        checking_data, checking_data_output = zip(*[(self.input_data[i], self.correct_outputs_nn_format[i]) for i in
                                                    random_range[int(self.no_inputs * training_data_fraction):]])
        print("done")

        # split training data into batches, e.g., [1, 5, 7, 6, 3, 9, 5] with batch size 3 will result in [[1, 5, 7], [6, 3, 9], [5]]
        print("creating batches ...  ", end='')
        batches = np.array([training_data[i * batch_size:(i + 1) * batch_size] for i
                            in range((len(training_data) + batch_size - 1) // batch_size)])
        batches_output = np.array([training_data_output[i * batch_size:(i + 1) * batch_size] for i
                                   in range((len(training_data) + batch_size - 1) // batch_size)])
        print("done")

        for batch, batch_output in zip(batches, batches_output):
            print("training batch ...  ", end='')
            self.update_weights_and_biases(batch, batch_output)
            print("done")

        print("Results:")
        self.check_accuracy(training_data, training_data_output, checking_data, checking_data_output)


neural_network = NN('./processed_data.pkl', 14, 20, 50, 5, 5)
neural_network.train(10, 0.8)
