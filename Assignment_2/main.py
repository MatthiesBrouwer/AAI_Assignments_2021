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

        #print("Data: {}".format(self.datalist[indices]))

        return self.datalist[indices]

    def getLabelsByIndices(self, indices):
        # Initialize the guard clauses
        if (np.max(indices) > len(self.datalabels)):
            reportError("Dataset", "getLabelsByIndices", "Requested index higher than available datalabels")

        return self.datalabels[indices]


    def getLabeledDataByIndices(self, indices):
        # Initialize the guard clauses
        if (np.max(indices) > len(self.datalabels)):
            reportError("Dataset", "getLabeledDataByIndices", "Requested index higher than available datalabels")
        if (np.max(indices) > len(self.datalist)):
            reportError("Dataset", "getLabeledDataByIndices", "Requested index higher than available datalist")

        #Get the data and labels using the indices
        data = self.getDataByIndices(indices)
        labels = self.getLabelsByIndices(indices)

        labeled_data = []

        for index in range(0, len(indices)):
            labeled_data.append([labels[index], data[index]])

        #Zip the data and labels arrays together
        return labeled_data


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

        # Make sure the dataset is normalized
        dataset.normalizeData()

        #Evaluate the values for k using the k-means clustering algorithm
        for k_value in range(k_start, k_end + 1):

            #Obtain an array containing centroid indexes at the index corresponding to their assclusters and their intra_distance

            #Cluster the dataset and obtain an array where each index corresponds to a piece of data and the value
            #contained at the index to the centroid they were assigned to
            clustered_datapoint_indices__total = self.clusterDataset(k_value, dataset, 4)

            print(clustered_datapoint_indices__total)
            clusters_intradistance = self.calcTotalCentroidIntraDistance(k_value, clustered_datapoint_indices__total, dataset)
            print("Total distance: {}".format(clusters_intradistance))

            clusters_seasons = self.getClusterSeasons(k_value, clustered_datapoint_indices__total, dataset)

            print("Cluster seasons: {}".format(clusters_seasons))

    def clusterDataset(self, k_value, dataset, iterations = 1):
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
        if (iterations < 1):
            self.reportError("SeasonClusterer", "clusterDataset", "At least one iteration should be tried")

        #Initialize a variable to hold the best intradistance obtained during the k-means iterations
        intradistance_min = 9223372036854775807

        #Also initialize a variable to hold the assigned clusters array corresponding to the iterations with
        #the best intradistance
        assigned_centroids__best = []

        #Try multiple iterations of k-means, using different centroid starting positions each time
        #and return the clusters with the lowest total intra distance
        for iteration in range(0, iterations):

            # Pick k random points from the dataset to serve as starting positions for the centroids
            centroids = random.choices(dataset.datalist, k=k_value)

            #The following portion assigns the points of data to their closest centroid
            #and afterwards reassigns each centroid to the mean of all points of data assigned to them.
            #It continuously does this until no changes in datapoint assignments are occuring.

            #An array where each index corresponds to the index of a datapoint in the dataset and each value to
            #the index of their assigned centroid.
            assigned_centroids = np.zeros(len(dataset.datalist))

            for iteration in range (0, 100000):
                #if no change occurs, stop iterating
                no_change = True
                #print("Centroids: {}".format(centroids))
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


            #Calculate the intradistance of the current cluster formations. Store it and the assigned centroids array
            #if it's the intradistance is lower than the previous lowest
            intradistance_current = self.calcTotalCentroidIntraDistance(k_value, assigned_centroids, dataset)
            if(intradistance_current < intradistance_min):
                intradistance_min = intradistance_current
                assigned_centroids__best = assigned_centroids


        #return the assigned centroids
        return assigned_centroids__best

        """
        #For each centroid, collect an array containing tuples of datalabel and datapoint for each point of data
        #assigned to them.

        #print("Assigned: {}".format(assigned_centroids))

        for centroid_index in range(0, len(centroids)):
            # Get an array of the indexes all datapoints assigned to the centroid at the current index

            #Skip this step if no datapoints were assigned to a centroid
            if not np.isin(centroid_index, assigned_centroids):
                continue

            #Get the indexes of the datapoints assigned to the centroid
            assigned_datapoints_indexes = np.where(assigned_centroids == centroid_index)[0]

            #Get an array of the datalabels and datapoints of all datapoints assigned to the centroid
            labeled_datapoints = dataset.getLabeledDataByIndices(assigned_datapoints_indexes)
            print(labeled_datapoints)


        #Return a tuple. The first element contains a nested array of centroids, where their index represents the
        #centroids index. The second element contains an array containing the assigned_centroids array.


        """


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

    def updateCentroids(self, k_value, assigned_centroid_indices, dataset):
        #Wants an array where the indexes relate to the
        #datapoint indexes and the data contained to which cluster they were assigned to.
        #This is represented as the index in the cluster array
        #It returns a new array containing the new centroids

        # Initialize the guard clauses
        if (assigned_centroid_indices.max(axis=0) > (k_value - 1)):
            self.reportError("SeasonClusterer", "updateCentroids", "Value of k is lower than amount of assigned centroids")

        #For each value of k representing the amount of centroids used, recalculate their centroid mean
        centroids = []

        for centroid_index in range(0, k_value):

            #Skip the recalculation if no datapoints were assigned to a centroid
            if not np.isin(centroid_index, assigned_centroid_indices):
                continue

            #Get an array of the indexes all datapoints assigned to the centroid at the current index
            assigned_datapoints_indices = np.where(assigned_centroid_indices == centroid_index)

            #Get an array of the values of all datapoints assigned to the centroid at the current index
            assigned_datapoints = dataset.getDataByIndices(assigned_datapoints_indices)

            # Calculate the mean of all assigned indexes
            centroids.append(assigned_datapoints.mean(axis = 0))

        return centroids

    def calcTotalCentroidIntraDistance(self, k_value, datapoint_clusters, dataset):

        #A value to hold the total intradistance between all clusters
        intradistance_clusters = 0

        # Calculate the intra distance of each cluster using their indices
        for centroid_index in range(0, k_value):

            # Skip this step if no datapoints were assigned to a centroid
            if not np.isin(centroid_index, datapoint_clusters):
                continue

            # Get the indices of the datapoints assigned to the centroid
            datapoint_indices__partial = np.where(datapoint_clusters == centroid_index)[0]

            # Get the values of the datapoints assigned to the centroid using their indices
            clustered_datapoint_values__partial = dataset.getDataByIndices(datapoint_indices__partial)

            # Calculate the intra-distance for the current cluster and add it to the total
            intradistance_clusters += self.calcCentroidIntraDistance(clustered_datapoint_values__partial)


        return intradistance_clusters

    def calcCentroidIntraDistance(self, datapoints):
        #a function that calculates the intra distance between datapoints assigned to a cluster using an
        #array of indices

        # Initialize the guard clauses
        if (len(datapoints) == 0):
            reportError("SeasonClusterer", "calcCentroidIntraDistance", "No datapoints provided")

        #Get the position of the centroid, which is at the mean of all datapoints
        centroid = datapoints.mean(axis = 0)

        def sumIntraDistances(index = 0):
            #This function recursively calculates the distance from an assigned point to the centroid and adds it
            #to the ones previously calculated until all have been calculated

            #If all distances have been calculated, simply return 0
            if index == len(datapoints):
                return 0

            #Calculate the squared euclidian distance between a point and the centroid
            current_distance = np.square(self.calcEuclideanDistance(centroid, datapoints[index]))

            return current_distance + sumIntraDistances(index + 1)

        #Return the result from the recursive function above
        return sumIntraDistances()


    def getClusterSeasons(self, k_value, datapoint_clusters, dataset):

        #An array to hold the season most common among datapoints assigned per cluster. The index of each cluster
        #season corresponds to the cluster index
        cluster_seasons = []

        for centroid_index in range(0, k_value):

            # Skip this step if no datapoints were assigned to a centroid
            if not np.isin(centroid_index, datapoint_clusters):
                cluster_seasons.append("NONE")
                continue

            # Get the indices of the datapoints assigned to the centroid
            datapoint_indices__partial = np.where(datapoint_clusters == centroid_index)[0]

            # Get an array of the datalabels of all datapoints assigned to the centroid at the current index
            assigned_datapoint_labels = dataset.getLabelsByIndices(datapoint_indices__partial)


            cluster_seasons.append(max(set(assigned_datapoint_labels), key = lambda season : ((assigned_datapoint_labels == season).sum())))

        return cluster_seasons

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
dataset_2000 = DataSet(2001, "dataset_2001.csv")

#Create the season clusterer
season_clusterer = SeasonClusterer()

#Cluster the dataset using the season clusterer
season_clusterer.evaluate(dataset_2000, 4, 4)
