import numpy as np

class DataSet:
    """
    This class holds the data and seasonlabel for weather data in the following format:
    [YYYYMMDD][FG][TG][TN][TX][SQ][DR][RH]
    Example:
        20010102;60;87;56;112;27;9;3

    The seasons are determined based on the date of measurement, and are stored in
    a seperate numpy array but are in the same order as the array containing the
    actual measurements
    """
    def __init__(self, year, filepath):

        #Read the data from the file, excluding the first attribute containing the date of measurement
        self.datalist = np.genfromtxt(filepath,
                                 delimiter=";",
                                 usecols=[1, 2, 3, 4, 5, 6, 7],
                                 converters={5: lambda s: 0 if s == b"-1" else float(s),
                                             7: lambda s: 0 if s == b"-1" else float(s)})

        #Read the dates from the file
        dates = np.genfromtxt(filepath, delimiter=";", usecols=[0])

        self.datalabels = np.array([])

        #Create a yearcode corresponding to the year when the measurements were taken.
        yearcode = year * 10000

        for label in dates:
            if label < yearcode + 301:
                self.datalabels = np.append(self.datalabels, "winter")
            elif yearcode + 301 <= label < yearcode + 601:
                self.datalabels = np.append(self.datalabels, "spring")
            elif yearcode + 601 <= label < yearcode + 901:
                self.datalabels = np.append(self.datalabels, "summer")
            elif yearcode + 901 <= label < yearcode + 1201:
                self.datalabels = np.append(self.datalabels, "fall")
            else:  # from 01-12 to end of year
                self.datalabels = np.append(self.datalabels, "winter")


    def getNormalized(self, range_min = 0, range_max = 1):
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

    def getDataByIndexes(self, indexes):
        pass

    def getLabelsByIndexes(self, indexes):
        pass

    def getLabeledDataByIndexes(self, indexes):
        pass

    def normalizePoint(self, point, range_min = 0, range_max = 1):
        """
        A function that normalizes a point in accordance to the
        details of the dataset. The technique used is called "Feature Scaling"
        https://en.wikipedia.org/wiki/Feature_scaling
        It scales each feature vector in a dataset based on
        their respective collumn max and min values.
        The formula is:
        x' = range_min + (((x - min(x)) * (range_max - range_min))) / (max(x) - min(x))

        :param point: The point to be normalized in accordance to the dataset
        :param range_min: The min value for the normalization
        :param range_max: the end value for k
        :return: the normalized point
        """
        nom = (point - self.datalist.min(axis=0)) * (range_max - range_min)
        denom = self.datalist.max(axis=0) - self.datalist.min(axis=0)
        denom[denom == 0] = 1
        point_normalized = range_min + nom / denom

        #print(point_normalized)
        return point_normalized


class SeasonClusterer:
    def __init__(self):

        pass

    def evaluate(self, dataset, k_start, k_end):
        pass

    def clusterDataset(self, k, dataset):
        #returns cluster details in array of tuples [mostcommonseason, intradistance]
        #calls assignDataPoint() to get the new cluster index. If one is diferent than
        #last call it loops. If none change it finishes
        pass

    def assignDatapoint(self, datapoint, cluster_centroids):
        #Uses the clusters to determine which cluster the datapoints belongs to and returns it
        pass

    def updateCentroids(self, assigned_indexes):
        #Wants an array where the indexes relate to the
        #datapoint indexes and the data contained to which cluster they were assigned to.
        #This is represented as the index in the cluster array
        #It returns a new array containing the new centroids
        pass

    def calcCentroidIntraDistance(self, assigned_indexes, cluster_centroids):
        #a function that calculates the intra distance between all clusters and their
        #assigned datapoints
        pass

    def plotScreeGraph(self, intradistances):
        #A function that uses the provided array containing tuples of [k_value, intradistance]
        #to plot a scree array




