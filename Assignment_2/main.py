import numpy as np
import random

TEST_PRINTS = True


def reportError(class_name, function_name, message):
    raise Exception(class_name, ", ", function_name, " : ", message)


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

    def normalizeData(self, range_min = 0, range_max = 1):
        nom = (self.datalist - self.datalist.min(axis=0)) * (range_max - range_min)
        denom = self.datalist.max(axis=0) - self.datalist.min(axis=0)
        denom[denom == 0] = 1
        self.datalist = range_min + nom / denom

    def getDataByIndices(self, indices):
        """
        A function that returns an array containing the points of data at the provided indexes
        Args:
            indices: The indices of the desired points of data

        Returns: The data at the provided indexes
        """

        #Initialize the guard clauses
        if (np.max(indices) > len(self.datalist)):
            reportError("Dataset", "getDataByIndices", "Requested index higher than available data")

        return self.datalist[indices]

    def getLabelsByIndices(self, indices):
        pass

    def getLabeledDataByIndices(self, indices):
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
        #Initialize guard clauses
        if(k_end < k_start or k_start <= 1 or k_end <= 1):
            self.reportError("SeasonClusterer", "evaluate", "Invalid k bounderies")
        if(len(dataset.datalist) == 0):
            self.reportError("SeasonClusterer", "evaluate", "Empty dataset provided")
        if (k_end > len(dataset.datalist)):
            self.reportError("SeasonClusterer", "evaluate", "Value for k is larger size of dataset")

        print("===========================")
        print("----Starting evaluation----")
        print("k start:\t{}".format(k_start))
        print("k end:\t{}\n\n".format(k_end))

        #Evaluate the values for k using the k-means clustering algorithm
        for k_value in range(k_start, k_end + 1):

            #Obtain the clusters and their intra_distance
            self.clusterDataset(k_value, dataset)


    def clusterDataset(self, k_value, dataset):
        #returns cluster details in array of tuples [mostcommonseason, intradistance]
        #calls assignDataPoint() to get the new cluster index. If one is diferent than
        #last call it loops. If none change it finishes

        # Initialize guard clauses
        if (k_value <= 1):
            self.reportError("SeasonClusterer", "clusterDataset", "Invalid value for k")
        if (len(dataset.datalist) == 0):
            self.reportError("SeasonClusterer", "clusterDataset", "Empty dataset provided")
        if(k_value > len(dataset.datalist)):
            self.reportError("SeasonClusterer", "clusterDataset", "Value for k is larger size of dataset")

        #Make sure the dataset is normalized
        dataset.normalizeData()

        # Pick k random points from the dataset to serve as starting positions for the centroids
        centroids = random.choices(dataset.datalist, k=k_value)

        #The following portion assigns the points of data to their closest centroid
        #and afterwards reassigns each centroid to the mean of all points of data assigned to them.
        #It continuously does this until no changes in datapoint assignments are occuring.

        #Declare a variable that determines wether no datapoint's assigned centroid were changed during
        #this rendition. If it's still true after all centroids were assigned, it breaks the loop
        no_change = True

        #An array where each index corresponds to the index of a datapoint in the dataset and each value to
        #the index of their assigned centroid.
        assigned_centroids = np.zeros(len(dataset.datalist))

        max_iterations = 100000

        for iteration in range (max_iterations):
            #if no change occurs, stop iterating
            no_change = True
            print("Centroids: {}".format(centroids))
            #iterate over all point of data
            for datapoint_index in range(0, len(dataset.datalist)):

                #Assign the datapoint at the current index and store it at the correct index
                assigned_centroid_new = self.assignDatapoint(dataset.datalist[datapoint_index], centroids)

                if(assigned_centroid_new != assigned_centroids[datapoint_index]):
                    no_change = False
                    assigned_centroids[datapoint_index] = assigned_centroid_new


            #If no changes occcured during the entire process, break the loop
            if(no_change):
                break

            #Otherwise, recalculate the centroid positions using the newly assigned means and redo the previous process
            centroids = self.updateCentroids(k_value, assigned_centroids, dataset)



        print(assigned_centroids)






    def assignDatapoint(self, datapoint, centroids):
        #Uses the clusters to determine which cluster the datapoints belongs to and returns it

        #Initialize the guard clauses
        if(len(datapoint) == 0):
            self.reportError("SeasonClusterer", "assignDatapoint", "Empty datapoint provided: {}".format(datapoint))
        if(len(centroids) == 0):
            self.reportError("SeasonClusterer", "assignDatapoint", "Empty centroid array provided: {}".format(centroids))

        #A variable representing the minimal distance between the datapoint and a centroid
        distance_minimum = 999999999
        #A variable representing the index of the nearest centroid
        nearest_centroid_index = -1

        #Find the index of the nearest centroid
        for centroid_index in range(0, len(centroids)):
            #Get the euclidian distance between the datapoint and the current centroid
            distance = self.calcEuclideanDistance(datapoint, centroids[centroid_index])

            if(distance < distance_minimum):
                distance_minimum = distance
                nearest_centroid_index = centroid_index

        #Return the index of the nearest centroid
        return nearest_centroid_index

    def updateCentroids(self, k_value, assigned_centroid_indexes, dataset):
        #Wants an array where the indexes relate to the
        #datapoint indexes and the data contained to which cluster they were assigned to.
        #This is represented as the index in the cluster array
        #It returns a new array containing the new centroids

        # Initialize the guard clauses
        if (assigned_centroid_indexes.max(axis=0) > (k_value - 1)):
            self.reportError("SeasonClusterer", "updateCentroids", "Value of k is lower than amount of assigned centroids")

        #For each value of k representing the amount of centroids used, recalculate their centroid mean
        centroids = []

        for centroid_index in range(0, k_value):

            #Get an array of the indexes all datapoints assigned to the centroid at the current index
            assigned_datapoints_indexes = np.where(assigned_centroid_indexes == centroid_index)

            #Get an array of the values of all datapoints assigned to the centroid at the current index
            assigned_datapoints = dataset.getDataByIndices(assigned_datapoints_indexes)

            # Calculate the mean of all assigned indexes
            centroids.append(assigned_datapoints.mean(axis = 0))

        return centroids

    def calcCentroidIntraDistance(self, assigned_indexes, cluster_centroids):
        #a function that calculates the intra distance between all clusters and their
        #assigned datapoints

        pass

    def calcEuclideanDistance(self, point_1, point_2):
        """
        his function calculates the distance between two normalized points
        :param point_1: The first point
        :param point_2: The second point
        :return: The distance between the two points
        """

        distance = np.linalg.norm(point_2 - point_1)
        return distance

    def plotScreeGraph(self, intradistances):
        #A function that uses the provided array containing tuples of [k_value, intradistance]
        #to plot a scree array
        pass

#Set the random seed to 0
random.seed(0)


#Create a dataset to cluster
dataset_2000 = DataSet(2000, "dataset_2000.csv")

#Create the season clusterer
season_clusterer = SeasonClusterer()

#Cluster the dataset using the season clusterer
season_clusterer.evaluate(dataset_2000, 4, 4)


