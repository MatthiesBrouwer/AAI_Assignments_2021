
import numpy as np
import math

#Define a global ZETA value
ZETA = 0.1

def sigmoid(value):
	"""A function that returns the sigmoid version of it.
	it performs the sigmoid function (1 / (1 + e^-value))"""

	if (value >= 0):
		#return (1 / (1 + (math.e ** (-value))))

		return (1 / (1 + math.exp(-value)))
		#return math.tanh(value)
	else:
		return (math.exp(value) / (1 + math.exp(value)))

class IrisDataset:
	def __init__(self, filepath : str, attribute_mnt__init, unique_labels__init):

		#Read the data from the file, exluding the last attribute, as it contains the name of the iris flower
		self.datalist = np.genfromtxt(filepath, 
					delimiter=",",
					usecols=[0, 1, 2, 3])

		#Read the datalabels from the file in the last attribute
		self.labellist = np.genfromtxt(filepath, 
					delimiter=",",
					usecols=[4],
					dtype="str")

		#Store the attribute amount and label amound
		self.attribute_mnt = attribute_mnt__init
		self.unique_labels = unique_labels__init


	def getNormalized(self, range_min : int = 0, range_max : int = 1) -> np.ndarray :
		"""
		A function that returns the stored dataset list in normalized form.
		The technique used is called "Feature Scaling"
		https://en.wikipedia.org/wiki/Feature_scaling
		It scales each feature vector in a dataset based on
		their respective collumn max and min values.
		The formula is:
		x' = range_min + (((x - min(x)) * (range_max - range_min))) / (max(x) - min(x))
		:param range_min: The min value for the normalization
		:param range_max: the end value for k
		:return: An normalized ndarray of the stored dataset
		"""
		nom = (self.datalist - self.datalist.min(axis=0)) * (range_max - range_min)
		denom = self.datalist.max(axis=0) - self.datalist.min(axis=0)
		denom[denom == 0] = 1
		datalist_normalized = range_min + nom / denom
		return datalist_normalized


	def getUniqueLabels(self):
		"""This function returns the unique labels among the labelist. 
		"""
		return self.unique_labels


		
class NeuralNetwork:
	"""
	This class represents a neural network. It contains multiple layers including an input and
	an output layer. It functions just like a normal neural network, namely using input on the
	throughput of its contained neurons to determine an output. This output can then be used to
	update the neural network towards the desired goal using backpropagation"""

	def __init__(self, network_shape__init : [int]):
		"""An initializer function
		This should create the layers and their neurons
		The shape of the neural network is provided using an array of integer.
		Each array index corresponds to a neuron layer and the integer to the amount of
		neurons it contains. The first and last indexes correspond to the input and output layer.
		:param network_shape__init: The shape of the network
		"""	
		#Initialize the guard clause
		if(len(network_shape__init) < 3):
			raise Exception("The amount layers cannot be lower than 3 (including input and output layer)")
		if(network_shape__init[0] < 1):
			raise Exception("The amount of neurons in the input layer can not be less than one")
		if(network_shape__init[-1] < 1):
			raise Exception("The amount of neurons in the output layer can not be less than one")



		#First, create an array that will contain the layers, and add the input layer to it.
		self._neuron_layers = []
		self._neuron_layers.append(InputNeuronLayer(network_shape__init[0]))
		
		
		#Next, add the hidden neuron layers to the array, providing each with a referene to the layer prior to them in the array
		#print(network_shape__init)
		for index in range(1, len(network_shape__init) - 1):
			self._neuron_layers.append(NeuronLayer(network_shape__init[index], self._neuron_layers[index - 1]))

		#Lastly, add the output neuron layer and give it a reference to the last added neuron layer
		self._neuron_layers.append(OutputNeuronLayer(network_shape__init[-1], self._neuron_layers[-1])) 
		

	def processInput(self, input_values : [int]):
		"""This function processes an array of input values using the neural network's structure
		and returns the output.
		:param input_values: the input value array
		"""

		#Initialize the guard clause
		if(len(input_values) != len(self._neuron_layers[0]._neurons)):
			raise Exception("Amount of input values does not match the amount of input neurons")

		#Update the outputs of the neurons in the input layer
		self._neuron_layers[0].setInputNeurons(input_values)

		
		#update the outputs of all neuron layers, starting with the output layer
		self._neuron_layers[-1].updateOutputs()

		#Gather the outputs and return the array
		output_array = []
		for neuron in self._neuron_layers[-1]._neurons:
			output_array.append(neuron.output)

		return output_array

	def trainNetwork(self, desired_output : [int], input_values : [int]):
		"""This function processes an array of input values using the neural network's structure and
		compares it to the desired ouput to update the neurons and their weights/biases
		:param desired_output: An array of values used to determine the delta of neurons
		:param input_values: the input value array
		"""
		#initialize the guard clause
		if(len(input_values) != len(self._neuron_layers[0]._neurons)):
			raise Exception("Amount of input values does not match the amount of input neurons")
		#First update the outputs of all neurons in the network using the input values
		self.processInput(input_values)
		
		#Calculate the deltas of the output neurons
		output_neuron_deltas = []
		for neuron_index in range(0, len(self._neuron_layers[-1]._neurons)):
			#Get the sigmoid output of the neuron
			output__sigmoid = sigmoid(self._neuron_layers[-1]._neurons[neuron_index].output)
			#Get the error
			error = desired_output[neuron_index] - output__sigmoid
		
			print("Error for neuron {}: {} - {} = {}".format(neuron_index, desired_output[neuron_index], output__sigmoid, error))
			#Multiply the error with the derived sigmoid of the neuron output to get the delta
			output__sigmoid_derived = output__sigmoid * (1 - output__sigmoid)
			output_neuron_deltas.append(output__sigmoid_derived * error)
		print("\n\n")

		#Use backpropagation to update the weights and biases of all previous neurons
		self._neuron_layers[-1].updateNeurons(output_neuron_deltas)	

		#print("AFTER UPDATING")
		#for layer in self._neuron_layers:
		#	print(layer)
		#print("\n\n")
		#print("\nOutputs:")
		#for output_neuron_index in range(0, len(self._neuron_layers[-1]._neurons)):
		#	print("Neuron {}: {}".format(output_neuron_index, self._neuron_layers[-1]._neurons[output_neuron_index]))
class NeuronLayer:
	"""
	This class contains the functionality of a neuron layer within the neural network. 
	It contains multiple neurons and their respective weights, and a link to the previous layer.
	"""
	def __init__(self, neuron_mnt__init : int, layer_prev__init : object = None):
		"""An initializer function
		It should store the previous layer and initialize the neurons contained in this layer
		providing them with the amount of neurons in the previous layer
		"""
		
		#Initialize the guard clause
		if(neuron_mnt__init < 1):
			raise Exception("Amount of neurons can not be less than one")

		#Save the previous layer
		self.layer_prev = layer_prev__init

		#Create an array that will contain all neurons in this layer
		self._neurons = []

		#Next, add all neurons to the list and instruct them on the amount of required
		#weigths using the amount of neurons in the previous layer. 
		#If no previous layer exists, instruct for 0 weights
		if(self.layer_prev != None):
			for index in range(0, neuron_mnt__init):
				self._neurons.append(Neuron(self.layer_prev.NeuronMnt()))
		else:
			for index in range(0, neuron_mnt__init):
				self._neurons.append(Neuron(0))		

	def __str__(self):
		"""A string representation of this class used for printing"""
		output_str = "\n============================\n"
		output_str += "Neuron layer:"

		for neuron in self._neurons:
			output_str += "\n" + neuron.__str__()
		return output_str + "\n"

	def __repr__(self):
		"""A representation function used for printing"""
		return self.__str__()
		
	def NeuronMnt(self):
		"""A getter function for the amount of neurons in this layer"""
		return len(self._neurons)

	def neuronOutputs(self):
		"""A getter function for the current outputs (Z) of the neurons in this layer"""
		outputs = []
		for neuron in self._neurons:
			outputs.append(neuron.output)
		return outputs

	def updateOutputs(self):
		"""This function calculates is used calculate the outputs (Z) of the neurons throughout the network.
		It first calculates the output of the previous layer and uses that to calculate its own, returning the result
		"""
		#First, update the previous layers outputs
		self.layer_prev.updateOutputs()

		#Get the outputs of the neurons in the previous layer
		layer_prev_outputs = self.layer_prev.neuronOutputs()

		#Update the neurons on the current layer using the outputs of the previous layer
		for neuron in self._neurons:
			neuron.updateOutput(layer_prev_outputs)

	def updateNeurons(self, neuron_deltas : [float]):
		"""This function is used during the backpropagation portion of the training process.
		Its parent has calculated the neuron deltas contained in this layer using the weights from 
		their neurons to the neurons in this layer. These are then
		used within this function to calculate the deltas of neurons on the previous layer.
		After a previous layer returns from this function the current layer instance will update
		the weights and biases of its neurons and return afterwards.
		"""

		#Initialize the guard clause
		if(len(neuron_deltas) != len(self._neurons)):
			raise Exception("Amount of neuron deltas does not match amount of neurons in this layer")

		#First, calculate the delta of each neuron in the previous layer using the
		#weight of each neuron contained in this layer to the neurons in the previous
		#layer. 
		
		#An array to hold the delta of each neuron in the previous layer
		neuron_deltas__prev = np.zeros(len(self.layer_prev._neurons))
		
		#for neuron_index in range(0, len(self._neurons)):
		#	for neuron_index__prev in range(0, len(self.layer_prev._neurons)):
		#		neuron_deltas__prev[neuron_index__prev] += (neuron_deltas[neuron_index] * self._neurons[neuron_index].weights[neuron_index__prev])

		for neuron_index__prev in range(0, len(self.layer_prev._neurons)):
			#Get the sum of all weighted deltas from neurons in this layer to the neuron in the prev layer
			weighted_delta_sum = 0
			for neuron_index in range(0, len(self._neurons)):
				weighted_delta_sum += (neuron_deltas[neuron_index] * self._neurons[neuron_index].weights[neuron_index__prev])

			#Multiply the weighted delta sum with the derived sigmoid output of the neuron in the previous layer to get its new delta
			output_prev__sigmoid = sigmoid(self.layer_prev._neurons[neuron_index__prev].output)
			neuron_deltas__prev[neuron_index__prev] = ((output_prev__sigmoid * (1 - output_prev__sigmoid)) * weighted_delta_sum)

		self.layer_prev.updateNeurons(neuron_deltas__prev)
			
		#Update the weights and biases of the neurons in this layer
		for neuron_index in range(0, len(self._neurons)):
			#For each neuron on the previous layer
			for neuron_index__prev in range(0, len(self.layer_prev._neurons)):
				#Update the weight from the neuron in this layer to the neuron in the previous layer	
				self._neurons[neuron_index].weights[neuron_index__prev] += (ZETA * neuron_deltas[neuron_index] * self.layer_prev._neurons[neuron_index__prev].output)
				
			#Update this neurons bias
			self._neurons[neuron_index].bias += (ZETA * neuron_deltas[neuron_index])

class InputNeuronLayer(NeuronLayer):
	"""This class represents an input neuron. It inherits its functionalities from
	the base Neuronlayer class. The real difference being that the output of the neurons
	in this layer can be set manualy to act as input to the neural network."""
	def __init__(self, neuron_mnt__init : int):
		super().__init__(neuron_mnt__init)

	def __str__(self):
		"""A representation function used for printing"""
		output_str = "\n============================\n"
		output_str += "Input neuron layer:"
		for neuron in self._neurons:
			output_str += "\n" + neuron.__str__()
		return output_str + "\n"

	def setInputNeurons(self, input_values):
		"""This function is used to set the input neurons for this layer
		to a provided value
		:param input_values: The new input values
		"""
		#Initialize the guard clause
		if(len(input_values) != len(self._neurons)):
			raise Exception("The amount of input values does not match the amount of input neurons")

		#Set the output of all inputs in this layer to the input values
		for neuron_index in range (0, len(self._neurons)):
			self._neurons[neuron_index].output = input_values[neuron_index]	

	def updateOutputs(self):
		"""This function overrides the updateOutputs function as there are no outputs to update.
		It simply returns"""
		return
	def updateNeurons(self, neuron_deltas):
		"""This function overrised the updateNeurons function as there are no weights to update
		it simply returns"""
		return


class OutputNeuronLayer(NeuronLayer):
	"""This class represents an output neuron. It inherits its functionalities from 
	the base NeuronLayer class. The real difference being that this layer can be used as
	starting point for the recursive backpropagation as well as the actual base input to output
	determination."""
	def __init__(self, neuron_mnt__init : int, layer_prev__init : NeuronLayer):
		super().__init__(neuron_mnt__init, layer_prev__init)

	def __str__(self):
		"""A representation function used for printing"""
		output_str = "\n============================\n"
		output_str += "Output neuron layer:"
		for neuron in self._neurons:
			output_str += "\n" + neuron.__str__()
		return output_str + "\n"

class Neuron:
	"""This class corresponds to a neuron in the neural network. They are fairly straightforward as
	they simply store values in containers to be used by the neuronlayer for calculations"""
	
	def __init__(self, weight_mnt : int):
		"""An initializer function"""
		
		#Set the guard clause
		if(weight_mnt < 0):
			raise Exception("The amount of weights cannot be less than 0")

		self.weights = np.random.uniform(low=0, high=1, size=(weight_mnt)) #The weights from this neuron to all neurons on the previous layer
		self.bias = np.random.uniform(low=0, high=1, size=None) #The bias of this neuron
		self.output = 0 #The last calculated output of this neuron

	def __str__(self):
		"""A representation function used for printing"""
		output_str = "Neuron:"
		output_str += "\n\t-Weights: "
		for weight in self.weights:
			output_str += "[" + str(weight) + "]"

		output_str += "\n\t-Bias: " + str(self.bias)		
		output_str += "\n\t-Last output: " + str(self.output)
		return output_str + "\n"

	def setWeights(self, weights_new : [float]): 
		"""This function sets the weight to a new array"""
		#Set the guard clause
		if(self.weights.shape is not weights_new.shape):
			raise Exception("The provided weight array is of the wrong shape.")

		self.weights = weights_new

	def getWeights(self): 
		"""This function returns the weight array"""
		return self.weights

	def updateOutput(self, output_child_neurons : [float]):
		"""This function sets the output"""
		#Initialize the guard clause
		if(len(output_child_neurons) < len(self.weights)):
			raise Exception("The provided amount of neuron outputs does not match the amount of weights in this neuron")

		#First, map the sigmoid formula (1 / (1 + e^-z)) to all outputs
		output_child_neurons__sigmoid = np.array([sigmoid(output) for output in output_child_neurons])

		#Next, multiple the sigmoid outputs with the stored weights to obtain the weighted sigmoid output		
		output_child_neurons__sigmoid__weighted = np.multiply(output_child_neurons__sigmoid, self.weights)

		#Sum the outputs, Add the bias, and update the output of this neuron
		self.output = np.sum(output_child_neurons__sigmoid__weighted) + self.bias



def unison_shuffled_copies(a, b):
	p = np.random.permutation(len(a))
	return a[p], b[p]

#Create a neural network class instance
dataset__training = IrisDataset("iris(training).data", 4, ["Iris-setosa", "Iris-versicolor", "Iris-virginica"])

normalized_datalist__training = dataset__training.getNormalized()

normalized_datalist__training, labellist__training = unison_shuffled_copies(normalized_datalist__training, dataset__training.labellist)

#Get the unique labels
unique_labels__training = dataset__training.getUniqueLabels()


neural_network = NeuralNetwork([dataset__training.attribute_mnt, 8, 5, 4, len(unique_labels__training)])



for itteration in range(0, 20):
	for index in range(0, len(normalized_datalist__training)):
		#Get the desired output neuron to fire
		#desired_label_index = unique_labels__training.index(dataset__training.labellist[index])
		desired_label_index = unique_labels__training.index(labellist__training[index])
	
		desired_output = np.zeros(len(unique_labels__training))
		desired_output[desired_label_index] = 1

		neural_network.trainNetwork(desired_output, normalized_datalist__training[index])


#Create a validation dataset
dataset__validation = IrisDataset("iris(validation).data", 4, ["Iris-setosa", "Iris-versicolor", "Iris-virginica"])

normalized_datalist__validation = dataset__validation.getNormalized()

#Get the unique labels
unique_labels__validation = dataset__validation.getUniqueLabels()

for index in range(0, len(normalized_datalist__training)):
	#Get the desired output neuron to fire
	desired_label_index = unique_labels__training.index(dataset__training.labellist[index])
	desired_output = np.zeros(len(unique_labels__validation))
	desired_output[desired_label_index] = 1
	output = neural_network.processInput(normalized_datalist__training[index])

	print("\nExpected: {}, {}, {}".format(desired_output[0], desired_output[1], desired_output[2]))
	print("Obtained: {}, {}, {}".format(output[0], output[1], output[2]))

