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
        self.no_types = no_types
        self.no_carriers = no_carriers
        self.no_outputs = no_types + no_carriers
        self.correct_outputs = df[df.columns[2:4]].to_numpy()
        # create numpy array of the correct output values for each neuron (number of images) X (number of output neurons)
        self.correct_outputs_nn_format = np.zeros((self.no_inputs, self.no_outputs))
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

        # Find derivatives with respect to all weights and biases for the first layer. As we will apply matrix-vector
        # multiplication that can couple variables, we need to keep them decoupled. For bias vector, this can be done
        # with matrix with diagonal being bias vector.
        bias_first_layer_decoupled = np.diag(self.bias_first_layer)
        dc_db_first_layer = (self.weights_second_layer.dot((bias_first_layer_decoupled.T *
                                                            NN.f_quadratic_derivative(first_linear_layer)).T).T *
                             dummy_derivative).T.sum(axis=1)
        # We need to avoid any coupling between weights. So we can iterate through each weight, find cost
        # derivative and update correct position in the matrix. Setup the matrix to store the derivative with respect
        # to each weight.
        dc_dw_first_layer = np.zeros(self.weights_first_layer.shape)
        # Dummy matrix stores derivatives of first linear layer with respect to first layer weights. It will be used
        # often for chain rule, so keep it outside the loop for efficiency.
        dummy_matrix = np.tile(input_data, (self.weights_first_layer.shape[0], 1))
        # Iterate over each weight
        for row in range(0, self.weights_first_layer.shape[0]):
            for column in range(0, self.weights_first_layer[1]):
                # Set up the decoupled vector to simplify calculations. It is filled with zeros, with exception of the
                # position where weight is allowed to change, and therefore cost derivative is not 0.
                decoupled_vector = np.zeros(self.weights_first_layer.shape[0])
                decoupled_vector[row] = dummy_matrix[row, column]
                # Apply chain rule to the decoupled vector and calculate the cost derivative with respect to weight
                dc_dw_dummy = np.sum(self.weights_second_layer.dot(decoupled_vector *
                                                                   NN.f_quadratic_derivative(first_linear_layer)) *
                                     dummy_derivative)
                # Update the dc/dw matrix
                dc_dw_first_layer[row, column] = dc_dw_dummy
        
        # dc_dw_first_layer = (self.weights_second_layer.dot(np.tile(input_data, (self.weights_first_layer.shape[0], 1)).T *
        #                      NN.f_quadratic_derivative(first_linear_layer).T).T * dummy_derivative).T

        # Find derivatives with respect to all weights and biases for the second layer. Fortunately no matrix/vector
        # is applied here, so derivatives will not be coupled.
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

    def check_accuracy(self, training_data, training_data_correct_output_nn_format, training_data_correct_output,
                       checking_data, checking_data_correct_output_nn_format, checking_data_correct_output):
        """ Determines the accuracy of the neural network and plots the comparison between the training data and checking data. """
        output_training = [self.nn_execution(input_data, output_data)[-2] for input_data, output_data in zip(training_data, training_data_correct_output_nn_format)]
        output_checking = [self.nn_execution(input_data, output_data)[-2] for input_data, output_data in zip(checking_data, checking_data_correct_output_nn_format)]

        output_training_choice = [(np.argmax(img[:self.no_types]), np.argmax(img[self.no_types:]) + self.no_types) for i, img in enumerate(output_training)]
        output_checking_choice = [(np.argmax(img[:self.no_types]), np.argmax(img[self.no_types:]) + self.no_types) for i, img in enumerate(output_checking)]

        # create lists for aircraft types and fleet carriers consisting of the number of correct and wrong answers
        types_results_training = np.zeros((self.no_types, 2))
        carriers_results_training = np.zeros((self.no_carriers, 2))
        for i, output in enumerate(training_data_correct_output):
            types_results_training[int(output[0])] += np.array([1, 0]) if output[0] == output_training_choice[i][0] else np.array([0, 1])
            carriers_results_training[int(output[1])] += np.array([1, 0]) if output[1] == output_training_choice[i][1] else np.array([0, 1])
        types_results_checking = np.zeros((self.no_types, 2))
        carriers_results_checking = np.zeros((self.no_carriers, 2))
        for i, output in enumerate(checking_data_correct_output):
            types_results_checking[int(output[0])] += np.array([1, 0]) if output[0] == output_checking_choice[i][0] else np.array([0, 1])
            carriers_results_checking[int(output[1])] += np.array([1, 0]) if output[1] == output_checking_choice[i][1] else np.array([0, 1])

        # some pyplot stuff
        bars_types_training = [aircraft_type[0] / sum(aircraft_type) * 100 for aircraft_type in types_results_training]
        bars_carrier_training = [carrier[0] / sum(carrier) * 100 for carrier in carriers_results_training]
        bars_types_checking = [aircraft_type[0] / sum(aircraft_type) * 100 for aircraft_type in types_results_checking]
        bars_carrier_checking = [carrier[0] / sum(carrier) * 100 for carrier in carriers_results_checking]

        type_labels = ['A319', 'A320', 'A318', '190', 'A321', '747', 'A330', 'A350', '757', '737', '787', '170', '767', '777']
        carrier_labels = ['EgyptAir', 'Suparna Airlines', 'China Eastern Airlines', 'easyJet', 'Delta Air Lines', 'Corendon Dutch Airlines',
                          'Romanian Air Transport', 'Garuda Indonesia', 'AnadoluJet', 'Air Arabia Maroc', 'KLM', 'Saudi Arabian Airlines',
                          'Aer Lingus', 'Emirates', 'Air China Cargo', 'Blue Air', 'Titan Airways', 'Air France', 'Aeroflot', 'Alitalia']

        x1 = np.arange(len(type_labels))  # the label locations
        x2 = np.arange(len(carrier_labels))
        width = 0.35  # the width of the bars

        fig, (ax1, ax2) = plt.subplots(1, 2, sharey='row')
        ax1.bar(x1 - width / 2, bars_types_training, width, label='Training data')
        ax1.bar(x1 + width / 2, bars_types_checking, width, label='Checking data')
        ax2.bar(x2 - width / 2, bars_carrier_training, width, label='Training data')
        ax2.bar(x2 + width / 2, bars_carrier_checking, width, label='Checking data')

        ax1.set_ylabel('Accuracy [%]')
        ax1.set_xticks(x1)
        ax1.set_xticklabels(type_labels)
        ax1.legend()
        ax2.set_xticks(x2)
        ax2.set_xticklabels(carrier_labels)
        ax2.legend()

        fig.tight_layout()
        plt.show()

    def train(self, batch_size: int, training_data_fraction: float):
        """ Splits data in training and checking data, creates batches, trains the neural network, and checks the accuracy. """
        # split input data into training data and checking data based on the training data fraction
        print("splitting data ...  ", end='')
        random_range = random.sample(range(self.no_inputs), self.no_inputs)
        training_data, training_data_output, training_data_output_nn_format = zip(*[(self.input_data[i], self.correct_outputs[i], self.correct_outputs_nn_format[i]) for i in random_range[:int(self.no_inputs * training_data_fraction)]])
        checking_data, checking_data_output, checking_data_output_nn_format = zip(*[(self.input_data[i], self.correct_outputs[i], self.correct_outputs_nn_format[i]) for i in random_range[int(self.no_inputs * training_data_fraction):]])
        print("done")

        # split training data into batches, e.g., [1, 5, 7, 6, 3, 9, 5] with batch size 3 will result in [[1, 5, 7], [6, 3, 9], [5]]
        print("creating batches ...  ", end='')
        batches = np.array([training_data[i * batch_size:(i + 1) * batch_size] for i in range((len(training_data) + batch_size - 1) // batch_size)])
        batches_output = np.array([training_data_output_nn_format[i * batch_size:(i + 1) * batch_size] for i in range((len(training_data) + batch_size - 1) // batch_size)])
        print("done")

        for batch, batch_output in zip(batches, batches_output):
            print("training batch ...  ", end='')
            self.update_weights_and_biases(batch, batch_output)
            print("done")

        print("Results:")
        self.check_accuracy(training_data, training_data_output, training_data_output_nn_format, checking_data, checking_data_output, checking_data_output_nn_format)


neural_network = NN('./processed_data.pkl', 14, 20, 50, 5, 5)
neural_network.train(10, 0.8)
