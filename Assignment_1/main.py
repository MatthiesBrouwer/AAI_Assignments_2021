import numpy as np
import matplotlib.pyplot as plt

class SeasonIdentifier:
    """
    A class used to identify the seasons based on weather data for a specific year.
    Is uses data stored in DataSet class instances as containers to determine
    when seasons start and end using a K-nearest neighbours algorithm.
    """

    def __init__(self):
        """
        An initialization function
        """
        self.dataset_dict = {}  # The dataset containing the YearDataset instances


    def printData(self):
        """
        A function to print all data contained in all datasets
        :return:
        """
        for year in self.dataset_dict:
            print("Year\t\t\t: " + str(year))
            print(self.dataset_dict[year])



    class DataSet:
        """
        A class used to hold weather data for a specific year.
        It reads the data and the corresponding labels from provided files
        and stores them to be used by the SeasonIdentifier class.
        """

        def __init__(self, filepath):

            # Read the data from the file, excluding the first collumn as this contains
            # the data labels. Also, if the data in collumn 5 and 7 is <
            self.data_list = np.genfromtxt(filepath,
                                           delimiter=";",
                                           usecols=[1, 2, 3, 4, 5, 6, 7],
                                           converters={5: lambda s: 0 if s == b"-1" else float(s),
                                                       7: lambda s: 0 if s == b"-1" else float(s)})

            dates = np.genfromtxt(filepath, delimiter=";", usecols=[0])

            self.data_labels = []
            for label in dates:
                if label < 20000301:
                    self.data_labels.append("winter")
                elif 20000301 <= label < 20000601:
                    self.data_labels.append("spring")
                elif 20000601 <= label < 20000901:
                    self.data_labels.append("summer")
                elif 20000901 <= label < 20001201:
                    self.data_labels.append("fall")
                else:  # from 01-12 to end of year
                    self.data_labels.append("winter")

        def __repr__(self):
            """
            A function to print this instance and all it's data
            :return:
            """
            repr_str = "Dataset class instance\n"
            repr_str += "Days measured  : " + str(self.data_list.shape[0]) + "\n"
            repr_str += "Data amount    : " + str(self.data_list.size) + "\n"
            repr_str += "Data type      : " + str(self.data_list.dtype) + "\n"

            for index in range (0, len(self.data_list)):
                repr_str += self.data_labels[index] + " : " + str(self.data_list[index]) + "\n"
            return repr_str



        def getFeaturesMax(self):
            """
            A function that returns an array containing the max value of all data contained in the
            dataset

            :return: An array with the max for all data
            """

            #For each feature get the index of the feature vector where the max value was measured
            max_indices = np.argmax(self.data_list, axis=0)

            #Use these indices to extract the actual maximum values per feature
            max_values = []
            for feature_index in range (0, self.data_list.shape[1]):
                max_values.append(self.data_list[max_indices[feature_index]][feature_index])

            return max_values

        def getFeaturesMin(self):
            """
            A function that returns an array containing the min value of all data contained in the
            dataset

            :return: An array with the min for all data
            """
            # For each feature get the index of the feature vector where the min value was measured
            max_indices = np.argmin(self.data_list, axis=0)

            # Use these indices to extract the actual minimum values per feature
            max_values = []
            for feature_index in range(0, self.data_list.shape[1]):
                max_values.append(self.data_list[max_indices[feature_index]][feature_index])

            return max_values

        def getNormalized(self):

            # Calculate the minimal and maximum values for all features in all feature vectors
            min_feature_values = self.getFeaturesMax()
            max_feature_values = self.getFeaturesMin()

            # Calculate the range of all stored data
            feature_range = np.subtract(max_feature_values, min_feature_values)

            normalized = []
            for feature_vector in self.data_list:
                


    def addDataset(self, year, filepath):
        """
        A function that creates a new DataSet class instance and adds it to the
        dataset dictionary class attribute.

        Args:
            year (int): the year when the weather data was measured
            filepath (string): The path to the weather data .csv file
        """

        # Create a new DataSet instance and add it to the dataset dictionary
        self.dataset_dict[year] = self.DataSet(filepath)


    def getDistanceBetweenPoints(self, point_1, point_2, max_range, min_range):
        """
        This function calculates the distance between two normalized points
        :param point_1: The first point
        :param point_2: The second point
        :return: The distance between the two points
        """

        # Calculate the range of the data points
        feature_range = np.subtract(max_range, min_range)

        # Normalize the first point from which the distance will be calculated
        point_1_norm = np.divide(np.subtract(point_1, min_range), feature_range)

        # Normalize the second point to which the distance will be calculated
        point_2_norm = np.divide(np.subtract(point_2, min_range), feature_range)

        # Substract the first point from the second point
        data_cords = np.subtract(point_2_norm, point_1_norm)

        # Use pythagoras to calculate the distance between the two points
        distance = np.sum(list(map(lambda x: x * x, data_cords)))

        return distance


    def getDistanceToAllPoints(self, year, point):
        """
        This function calculates the distance from each data input in the dataset for a
        specified year from another data input using pythagoras.

        :param the year of the specific dataset:
        :param the point to which all distances are calculated:
        :return:
        """
        #Calculate the minimal and maximum values for all features in all feature vectors in the specified dataset
        min_feature_values = self.dataset_dict[year].getFeaturesMax()
        max_feature_values = self.dataset_dict[year].getFeaturesMin()

        # Calculate the range of all stored data
        feature_range = np.subtract(max_feature_values, min_feature_values)

        #Normalize the data point from which the distance will be calculated
        point_norm = np.divide(np.subtract(point, min_feature_values), feature_range)

        def calcDistances(index = 0):
            """
            A recursive function that calculates the distance from the provided reference point to all
            normalized points of data in a dataset for the specified year
            :param the index of the feature vector the function is to calculate the distance to:
            :return a list of all currently calculated distances:
            """
            #If the distance to all data vectors has been calculated, return None
            if(index >= len(self.dataset_dict[year].data_list)):
                return None


            current_distance = self.getDistanceBetweenPoints(point, self.dataset_dict[year].data_list[index], max_feature_values, min_feature_values)

            #Increment the index and get the next distance
            index += 1
            next_distance = calcDistances(index)

            # If the next next distance is not none, return it joined with the currently obtained distances. Otherwise
            # return only the current distance stored within a list
            return ([current_distance, *next_distance] if next_distance is not None else [current_distance])


        distances = calcDistances()

        #Zip the distances with the data labels in order to have the corresponding season per distance
        labeled_distances = np.dstack((self.dataset_dict[year].data_labels, distances))

        plt.plot(distances)

        plt.show()


        print(labeled_distances)









#Create a SeasonIdentifier class instance
season_identifier = SeasonIdentifier()

#Read the data from the year 2000. This is the training data
season_identifier.addDataset(2000, "dataset_2000.csv")

#Read the data from the year 2001. This is the validation data
season_identifier.addDataset(2001, "dataset_2001.csv")

#season_identifier.printData()

test_data = np.array([0, 0, 0, 0, 0, 0, 0])

season_identifier.getDistanceToAllPoints(2000, test_data)