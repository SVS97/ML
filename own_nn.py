import numpy
import scipy.special
import matplotlib.pyplot as plt


# defining of NN class
class neuralNetwork:

	# NN initializing
	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
		# defining count of nodes
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes

		#weight matrix
		self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
		self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

		# learning rate
		self.lr = learningrate

		#activation function (sigmoid)
		self.activation_function = lambda x: scipy.special.expit(x)
		pass

	# NN training
	def train(self, inputs_list, targets_list):
		# convert input list into 2D
		inputs = numpy.array(inputs_list, ndmin = 2).T
		targets = numpy.array(targets_list, ndmin = 2).T

		# input signals for hidden layer
		hidden_inputs = numpy.dot(self.wih, inputs)
		# output signals for hidden layer
		hidden_outputs = self.activation_function(hidden_inputs)

		# input signals for output layer
		final_inputs = numpy.dot(self.who,hidden_outputs)
		# output signals for output layer
		final_outputs = self.activation_function(final_inputs)

		# Errors of output layer
		output_errors = targets - final_outputs

		# Hidden errors
		hidden_errors = numpy.dot(self.who.T, output_errors)

		# update weights hidden-output layer
		self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

		# update weights input-hidden layer
		self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

		pass

	# NN inferience
	def query(self, inputs_list):
		# convert input list into 2D
		inputs = numpy.array(inputs_list, ndmin = 2).T

		# input signals for hidden layer
		hidden_inputs = numpy.dot(self.wih, inputs)
		# output signals for hidden layer
		hidden_outputs = self.activation_function(hidden_inputs)

		# input signals for output layer
		final_inputs = numpy.dot(self.who, hidden_outputs)
		# output signals for output layer
		final_outputs = self.activation_function(final_inputs)

		return final_outputs



# count of nodes
input_nodes = 784
hidden_nodes = 500
output_nodes = 10

# learning rate
learning_rate = 0.1

# create NN instance
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
#print(n.query([1.0, 0.5, -1.5]))

training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
#print(len(data_list))
#print(data_list[0])

epochs = 7

for e in range(epochs):
	print("Epoch: ", e+1)
	for record in training_data_list:


		all_values = record.split(',')
		#image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
		#plt.imshow(image_array, cmap = 'Greys', interpolation = 'None')
		#plt.show()

		inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
		#print(scaled_input)


		targets = numpy.zeros(output_nodes) + 0.01
		targets[int(all_values[0])] = 0.99
		#print(targets)
		n.train(inputs, targets)
		pass
	pass

test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

scorecard = []
for test_record in test_data_list:

	all_values = test_record.split(',')
	correct_label = int(all_values[0])
	print(correct_label, " truth marker")

	inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

	outputs = n.query(inputs)
	label = numpy.argmax(outputs)
	print(label, " network correspond")
	if (label == correct_label):
		scorecard.append(1)
	else:
		scorecard.append(0)
		pass
	pass

#print(scorecard)

scorecard_array = numpy.asarray(scorecard)
print("Precision = ", scorecard_array.sum() / scorecard_array.size)

