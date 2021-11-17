import numpy as np


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
            repr_str = ""
            repr_str += "Dataset class instance\n"
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



    def getDistanceToAllPoints(self, year, point):
        """
        This function calculates the distance from each data input in the dataset for a
        specified year from another data input using pythagoras.

        :param the year of the specific dataset:
        :param the point to which all distances are calculated:
        :return:
        """
        min_feature_values = self.dataset_dict[year].getFeaturesMax()
        max_feature_values = self.dataset_dict[year].getFeaturesMin()

        print("Min feature values: {}".format(min_feature_values))
        print("Max feature values: {}".format(max_feature_values))





    def printData(self):
        for year in self.dataset_dict:
            print(self.dataset_dict[year])



#Create a SeasonIdentifier class instance
season_identifier = SeasonIdentifier()

#Read the data from the year 2000
season_identifier.addDataset(2000, "dataset_2000.csv")

season_identifier.printData()

test_data = [0, 0, 0, 0, 0, 0, 0]

season_identifier.getDistanceToAllPoints(2000, test_data)