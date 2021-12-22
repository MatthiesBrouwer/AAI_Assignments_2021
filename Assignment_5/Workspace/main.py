import csv
import numpy as np
from collections.abc import Callable

#initialize type aliasses for the select, mutate, and crossover functions
select__typealias = Callable[[[tuple[float, [int]]], float], [[int]]]
mutate__typealias = Callable[[[int]], [int]]
crossover__typealias = Callable[[[int], [int]], [[int]]]



class DateKnightMatchmaker:
	"""
	This class uses a genetic algorithm to determine the perfect seating arangement
	for the upcomming date (k)night (pun intended) for the knights of the round table. 
	They each have some affection towards the other and should therefore be seated next 
	to someone whom they secretly have a crush on. 
	Therefore a genetic algorithm will try multiple positions to determine
	the (considered to be) best combination.

	Phenotype: A seating arangement of where which knight sits
	Genotype: An array of indexes, each index representing a knight


	An added feature of the class is the implementation of plug-and-play functions for
	the "select", "crossover", and "mutate" functionalities. This allows for multiple 
	functions to easily be changed during tested to select one that fits best. 
	The interface of these functions should be the following:
	
	- select:
		- params:
			graded_population : [tuple[float, [int]]] (The graded population, each with their fitness score)
			retain : float (The portion of the population to spawn offspring)
		- returns:
			new_population : [[int]] (The population selected as parents)

	- mutate:
		- params:
			individual : [int] (The individual to mutate)
		- returns:
			mutated_individual : [int] (The mutated individual)

	- crossover:
		- params:
			parent_1 : [int] (The first parent for crossover)
			parent_2 : [int] (The second parent for crossover)
		- returns:
			children : [[int]] (The child(ren) as a result of a crossover between the two parents) 
	"""

	def __init__(self, filepath__init : string): -> DateKnightMatchmaker
		"""
		An initializer function
		"""
		#Initialize placeholder members for select, crossover, and mutate functions
		self.__select = None
		self.__crossover = None
		self.__mutate = None

	def setSelectFunction(self, select__new : Callable[[[tuple[float, [int]]], float], [[int]]]): -> None
		"""
		A setter for the "select" function
		"""
		#Initialize the guard clause
		if not isinstance(select__new, select__typealias):
			raise Exception("Provided 'select' function has incorrect interface")
		self.__select = select__new

	
	def setMutateFunction(self, mutate__new : Callable[[[int]], [int]]): -> None
		"""
		A setter for the "mutate" function
		"""
		#Initialize the guard clause
		if not isinstance(mutate__new, mutate__typealias):
			raise Exception("Provided 'mutate' function has incorrect interface")
		
		self.__mutate = mutate__new

	
	def setCrossoverFunction(self, crossover__new : Callable[[[int], [int]], [[int]]]): -> None
		"""
		A setter for the "crossover" function
		"""
		#Initialize the guard clause
		if not isinstance(crossover__new, crossover__typealias):
			raise Exception("Provided 'crossover' function has incorrect interface")
		
		self.__crossover = crossover__new

	def genIndividual(self, min_val = 0 : int, max_val = 0 : int, unique = True : bool): -> [int]
		"""
		A function that generates an array of length (max - min) 
		containing either unique or non-unique  combinations of all values between
		min and max
		"""
		pass 

	def genPopulation(self, size : int): -> [[int]]
		"""
		A function that generates a nested array wherein each nested array
		contains a seating arangement represented by knight indexes
		"""
		pass


	def optimizePopulation(self, population : [[int]]): -> [[int]] 
		"""
		This function uses a genetic algorithm to determine the best seating
		arangement combination. It continues until a fitness threshold is obtained.
		"""
		pass

	def updatePopulation(self, population : [[int]]): -> [[int]]
		"""
		this function updates a population once using a genetic algorithem.
		"""
		pass


	def calcFitness__individual(self, individual : [int]): -> tuple[float, [int]]
		"""
		This function determines and returns the fitness of an individual
		"""
		pass

	def calcFitness__population(self, population : [[int]])
		"""
		This function determines and returns the fitness of each individual
		in a population. 
		"""
		pass


	
