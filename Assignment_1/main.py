import numpy as np
import matplotlib.pyplot as plt

TEST_PRINTS = True


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

        for index in range(0, len(self.data_list)):
            repr_str += self.data_labels[index] + " : " + str(self.data_list[index]) + "\n"
        return repr_str

    def getFeaturesMax(self):
        """
        A function that returns an array containing the max value of all data contained in the
        dataset

        :return: An array with the max for all data
        """

        # For each feature get the index of the feature vector where the max value was measured
        max_indices = np.argmax(self.data_list, axis=0)

        # Use these indices to extract the actual maximum values per feature
        max_values = []
        for feature_index in range(0, self.data_list.shape[1]):
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
        """
        A function that returns the stored dataset list in normalized form
        :return: An normalized ndarray of the stored dataset
        """

        #Get the sum of all features per feature vector
        row_sums = self.data_list.sum(axis=1)

        #Use these feature sums to normalize the enire data list
        new_matrix = self.data_list / row_sums[:, np.newaxis]
        #A = (self.data_list - np.mean(self.data_list)) / np.std(self.data_list)
        #print(A)

        normed_data = (self.data_list - self.data_list.min(0)) / self.data_list.ptp(0)
        print(normed_data)
        return normed_data


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
        self.trainingdata_dict = {}  # The dataset containing the YearDataset instances


    def printData(self):
        """
        A function to print all data contained in all datasets
        :return:
        """
        for year in self.trainingdata_dict :
            print("Year\t\t\t: " + str(year))
            print(self.trainingdata_dict[year])


    def addTrainingdata(self, year, filepath):
        """
        A function that creates a new DataSet class instance and adds it to the
        dataset dictionary class attribute.

        Args:
            year (int): the year when the weather data was measured
            filepath (string): The path to the weather data .csv file
        """

        # Create a new DataSet instance and add it to the dataset dictionary
        self.trainingdata_dict[year] = DataSet(filepath)


    def getDistanceBetweenPoints(self, point_1, point_2):
        """
        This function calculates the distance between two normalized points
        :param point_1: The first point
        :param point_2: The second point
        :return: The distance between the two points
        """

        # Substract the first point from the second point
        data_cords = np.subtract(point_1, point_2 )

        # Use pythagoras to calculate the distance between the two points
        #distance = np.sum(list(map(lambda x: x * x, data_cords)))
        distance = np.linalg.norm(data_cords)
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
        max_feature_values = self.trainingdata_dict[year].getFeaturesMax()
        min_feature_values = self.trainingdata_dict[year].getFeaturesMin()

        # Calculate the range of all stored data
        feature_range = np.subtract(max_feature_values, min_feature_values)

        #Normalize the data point from which the distance will be calculated
        #point_norm = np.divide(np.subtract(point, self.trainingdata_dict[year].data_list.min(0)), self.trainingdata_dict[year].data_list.ptp(0))
        point_norm = (point - self.trainingdata_dict[year].data_list.min(0)) / self.trainingdata_dict[year].data_list.ptp(0)
        #point_norm = (point - (self.trainingdata_dict[year].data_list + [point]).min(0)) / (self.trainingdata_dict[year].data_list + [point]).ptp(0)
        #normed_data = (self.data_list - self.data_list.min(0)) / self.data_list.ptp(0)


        #print("1: {}".format(np.subtract(max_feature_values, min_feature_values)))
        #print("2: {}".format(self.trainingdata_dict[year].data_list.ptp(0)))


        #Get a list of normalized feature vectors from the specified data set
        feature_vectors_norm = self.trainingdata_dict[year].getNormalized()
        print("before")
        def calcDistances(index = 0):
            """
            A recursive function that calculates the distance from the provided reference point to all
            normalized points of data in a dataset for the specified year
            :param the index of the feature vector the function is to calculate the distance to:
            :return a list of all currently calculated distances:
            """
            #If the distance to all data vectors has been calculated, return None
            if(index >= len(self.trainingdata_dict [year].data_list)):
                return None

            #Get the distance between the normalized point and the current feature vector
            #current_distance = self.getDistanceBetweenPoints(point_norm, feature_vectors_norm[index])
            current_distance = self.getDistanceBetweenPoints(point, self.trainingdata_dict[year].data_list[index])
            #Increment the index and get the next distance
            index += 1
            next_distance = calcDistances(index)

            # If the next next distance is not none, return it joined with the currently obtained distances. Otherwise
            # return only the current distance stored within a list
            return ([current_distance, *next_distance] if next_distance is not None else [current_distance])
        return calcDistances()


    def getNearestNeighbours(self, k, validation_vector):
        """
        A function that uses the calculated distance between the validation vector and all points
        in the training set, and returns the k nearest neighbouring feature vectors and their corresponding
        seasons.
        :param k: the value of k
        :param validation_vector: the vector from which the distance will be calculated
        :return: the k nearest neighbours with their corresponding seasons and distance
        """
        #Get the distance from the validation vector to all feature vectors in the trianing set
        distances = self.getDistanceToAllPoints(2000, validation_vector)

        #Get the indexes of smallest distances. Argpartition does not sort the array,
        #it guarantees that the k'th element is in sorted position and all smaller elements are before it.
        smallest_value_indexes = np.argpartition(distances, k)

        # Zip the distances with the data labels in order to have the corresponding season per distance
        labeled_distances = np.column_stack((self.trainingdata_dict[2000].data_labels, distances))

        #Extract the smallest distances and their corresponding seasons using the smallest value indexes
        nearest_neighbours = labeled_distances[smallest_value_indexes[:k]]

        if(not TEST_PRINTS):
            print("Function: getNearestNeighbours: ")
            print("Nearest neighbours: ")
            for distance in nearest_neighbours:
                print("\t", distance[0], " : ", distance[1])
            print()

        return nearest_neighbours

    def identifyFeatureVector(self, k, validation_vector):
        """
        this function determines the corresponding season to a validation vector using
        the k-nearest neighbour algorithm.
        :param k: the value for k
        :param validation_vector: the validation feature vector whom's season is to be determined
        :return: the determined season corresponding to the validation vector
        """
        #Get the k nearest neigbhbours to the validation vector
        nearest_neighbours = self.getNearestNeighbours(k, validation_vector)

        #Create a 2D array containin a season and the amount of neighbours corresponding to that season
        season_count = np.array([
            ["winter", np.count_nonzero(nearest_neighbours == "winter")],
            ["spring", np.count_nonzero(nearest_neighbours == "spring")],
            ["fall", np.count_nonzero(nearest_neighbours == "fall")],
            ["summer", np.count_nonzero(nearest_neighbours == "summer")],
        ])

        #Sort the array by second collumn
        season_count_sorted = season_count[np.argsort(season_count[:, 1])]


        if( TEST_PRINTS):
            print("Counted seasons: ")
            print("\tWinter: {}".format(season_count[0][1]))
            print("\tSpring: {}".format(season_count[1][1]))
            print("\tFall:   {}".format(season_count[2][1]))
            print("\tSummer: {}".format(season_count[3][1]))

        #Return the season which appeared the most often
        return season_count_sorted[-1][0]

    def evaluate(self, k, validation_set):
        """
        This function determines the succesrate of the k-nearest neighbour algorithm
        when using a specific value for k on a validation set
        :param k: the value for k
        :param validation_set: The set to be validated
        :return: The overall succesrate for all feature vectors in the validation set
        """
        #Get the season estimation using the K-neares neighbour algorithm

        correct_estimations = 0
        print(validation_set.data_labels)
        for index in range(0, len(validation_set.data_list)):
            estimated_season = self.identifyFeatureVector(k, validation_set.data_list[index])
            if(TEST_PRINTS):
                print("Estimated season: {}".format(estimated_season))
                print("Actual season:    {}".format(validation_set.data_labels[index]))
                print()
            if(estimated_season == validation_set.data_labels[index]):
                correct_estimations += 1

        succes_percentage = (correct_estimations / len(validation_set.data_list)) * 100

        print("Total validation feature vectors: {}".format(len(validation_set.data_list)))
        print("Correct estimations:              {}".format(correct_estimations))
        print("Succes percentage:                {}".format(succes_percentage))

        #estimated_season = self.identifyFeatureVector(k, validation_set.data_list[0])
        #print("Estimated season: {}".format(estimated_season))
        #print("Actual season:    {}".format(validation_set.data_labels[0]))





#Create a SeasonIdentifier class instance
season_identifier = SeasonIdentifier()

#Read the data from the year 2000. This is the training data
season_identifier.addTrainingdata(2000, "dataset_2000.csv")

#Read the data from the year 2001. This is the validation data
#season_identifier.addTrainingdata(2001, "dataset_2001.csv")
validation_set = DataSet("dataset_2001.csv")
feature_vectors_norm = validation_set.getNormalized()





#season_identifier.printData()

#test_data = np.array([0, 0, 0, 0, 0, 0, 0])

#season_identifier.getDistanceToAllPoints(2000, test_data)
#season_identifier.identifyFeatureVector(10, validation_set)
season_identifier.evaluate(57, validation_set)