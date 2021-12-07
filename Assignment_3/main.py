import numpy as np
import math



def sigmoid(value):
	"""A function that returns the sigmoid version of it.
	it performs the sigmoid function (1 / (1 + e^-value))"""

	if (value >= 0):
		return (1 / (1 + math.exp(-value)))
	else:
		return (math.exp(value) / (1 + math.exp(value)))




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
		print(network_shape__init)
		for index in range(1, len(network_shape__init) - 1):
			self._neuron_layers.append(NeuronLayer(network_shape__init[index], self._neuron_layers[index - 1]))

		#Lastly, add the output neuron layer and give it a reference to the last added neuron layer
		self._neuron_layers.append(OutputNeuronLayer(network_shape__init[-1], self._neuron_layers[-1])) 
		

		for layer in self._neuron_layers:
			print(layer)
		self._neuron_layers[-1].updateOutputs()


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

	def NeuronOutputs(self):
		"""A getter function for the current outputs (Z) of the neurons in this layer"""
		outputs = []
		for neuron in self._neurons:
			outputs.append(neuron.output)
		return outputs

	def updateOutputs(self):
		"""This function calculates is used calculate the outputs (Z) of the neurons throughout the network.
		It first calculates the output of the previous layer and uses that to calculate its own, returning the result
		"""
		 

	def updateNeurons(self, neuron_deltas : [float]):
		"""This function is used during the backpropagation portion of the training process.
		Its parent has calculated the neuron deltas using their weights, and these are then
		used within this function to calculate the deltas of neurons on the previous layer.
		After a previous layer returns from this function the current layer instance will update
		the weights and biases of its neurons and return afterwards.
		"""
		pass

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

	def updateOutputs(self):
		self._neurons[0].updateOutput([3, 2])
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
		print("Before: {}".format(output_child_neurons))
		print("After: {}".format(output_child_neurons__sigmoid))
		#Next, multiple the sigmoid outputs with the stored weights to obtain the weighted sigmoid output
		
		output_child_neurons__sigmoid__weighted = np.multiply(output_child_neurons__sigmoid, self.weights)
		print("\nAfter weights: {}".format(output_child_neurons__sigmoid__weighted))



#Create a neural network class instance
neural_network = NeuralNetwork([2, 3, 2, 1])


