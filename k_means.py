from gaussian_data import Gaussian_Data
import matplotlib.pyplot as plt
from skimage import io
import seaborn as sns
import pandas as pd
import numpy as np
import random


class K_Means():

    def __init__(self):
        self.point_assignments = []
        self.points = None
        self.results_dict = {}

    def get_score(self):
        euclidean_sum = 0
        for idx, row in self.points.iterrows():
            euclidean_sum += np.sqrt((row['point'] - row['cluster'])**2)
        return euclidean_sum

    def train(self, k, input_data, iterations):
        # Get list of input data
        self.points = pd.DataFrame(input_data)
        self.train_list = list(input_data)

        # Initialize cluster centers through random sampling
        self.cluster_centroids = random.sample(self.train_list, k)

        # Loop for number of iterations
        for iter in range(iterations):
            # Assign data points to closest cluster
            self.point_assignments = []
            for i in self.train_list:
                closest_distance = None
                closest_point = None
                for j in self.cluster_centroids:
                    distance = np.sqrt((i - j)**2)
                    if closest_distance is None:
                        closest_distance = distance
                        closest_point = j
                    elif distance < closest_distance:
                        closest_distance = distance
                        closest_point = j  
                self.point_assignments.append(closest_point)
            self.points['cluster'] = self.point_assignments

            # Update results dictionary
            self.results_dict.update({iter+1: {'centroids': self.cluster_centroids, 'score': self.get_score()}})

            # Don't update centroids on final iteration
            if iter == iterations-1:
                break

            # Update centroids for next iteration
            new_clusters = []
            for idx, j in enumerate(self.cluster_centroids):
                data_subset = self.points.loc[self.points['cluster'] == j]
                data_mean = data_subset['point'].mean()
                new_clusters.append(data_mean)
            self.cluster_centroids = new_clusters

    def get_results(self, labels):
        cluster_stats = {}
        for idx, i in enumerate(self.cluster_centroids):
            data_subset = self.points.loc[self.points['cluster'] == i]
            cluster_stats.update({idx+1: {'mean': data_subset['point'].mean(), 'sd': data_subset['point'].std()}})

        labelled_points = self.points
        labelled_points['mean'] = labels

        correct_count = 0
        total_count = 0
        for i in self.cluster_centroids:
            for j in range(100):
                data_subset = labelled_points.loc[labelled_points['cluster'] == i]
                sample_pair = data_subset.sample(n=2)
                if sample_pair.iloc[0]['mean'] == sample_pair.iloc[1]['mean']:
                    correct_count += 1
                total_count += 1

        return (cluster_stats, correct_count/total_count)

    def return_results_dict(self):
        return self.results_dict


if __name__ == "__main__":

    Data = Gaussian_Data()
    test_data = Data.read_gaussian(14)
    test_points = test_data[2]['point']

    K_Means = K_Means()
    K_Means.train(k=test_data[0], input_data=test_points, iterations=50)
    results = K_Means.get_results(test_data[2]['mean'])
    
    results_dict = K_Means.return_results_dict()
    score_list = []
    for key, val in results_dict.items():
        score_list.append(val['score'])
    x_list = list(range(len(score_list)))
    plt.plot(x_list, score_list)
    plt.show()
    print()



    # log_file = open('Results\\gaussian_exp1_results.txt', 'w')

    # # Get results for experiment 1 datasets with generators = 3
    # files = [1, 2, 3, 4]
    # sources = [2, 3, 6, 8]
    # for file in files:
    #     for source in sources:

    #         data = Gaussian_Data()
    #         test_data = data.read_gaussian(file)
    #         test_points = test_data[2]['point']

    #         k_means = K_Means()
    #         k_means.train(source, test_points, 50)
    #         results = k_means.get_results(test_data[2]['mean'])

    #         data_mean_list = []
    #         data_sd_list = []
    #         for key, val in test_data[1].items():
    #             data_mean_list.append(val['mean'])
    #             data_sd_list.append(val['sd'])
    #         data_means = ""
    #         for mean in data_mean_list:
    #             data_means += str(mean) + " "
    #         data_sd = ""
    #         for sd in data_sd_list:
    #             data_sd += str(sd) + " "

    #         pair_metric = str(round(results[1], 5))

    #         cluster_mean_list = []
    #         cluster_sd_list = []
    #         for key, val in results[0].items():
    #             cluster_mean_list.append(round(val['mean'], 5))
    #             cluster_sd_list.append(round(val['sd'], 5))
    #         cluster_means = ""
    #         for mean in cluster_mean_list:
    #             cluster_means += str(mean) + " "
    #         cluster_sd = ""
    #         for sd in cluster_sd_list:
    #             cluster_sd += str(sd) + " "

    #         output_string = "Sources: "+str(source)+", Data Means: ["+data_means+"], Data SD: ["+data_sd+"], Pair Metric: "+pair_metric+", Cluster Means: ["+cluster_means+"], Cluster SD: ["+cluster_sd+']\n'
    #         log_file.write(output_string)
        
    #     log_file.write('\n')

    # # Get results for other experiment 1 datasets
    # files = [5, 6, 7, 8, 9, 10, 11, 12]
    # for file in files:
    #     data = Gaussian_Data()
    #     test_data = data.read_gaussian(file)
    #     test_points = test_data[2]['point']

    #     k_means = K_Means()
    #     k_means.train(test_data[0], test_points, 50)
    #     results = k_means.get_results(test_data[2]['mean'])

    #     data_mean_list = []
    #     data_sd_list = []
    #     for key, val in test_data[1].items():
    #         data_mean_list.append(val['mean'])
    #         data_sd_list.append(val['sd'])
    #     data_means = ""
    #     for mean in data_mean_list:
    #         data_means += str(mean) + " "
    #     data_sd = ""
    #     for sd in data_sd_list:
    #         data_sd += str(sd) + " "

    #     pair_metric = str(round(results[1], 5))

    #     cluster_mean_list = []
    #     cluster_sd_list = []
    #     for key, val in results[0].items():
    #         cluster_mean_list.append(round(val['mean'], 5))
    #         cluster_sd_list.append(round(val['sd'], 5))
    #     cluster_means = ""
    #     for mean in cluster_mean_list:
    #         cluster_means += str(mean) + " "
    #     cluster_sd = ""
    #     for sd in cluster_sd_list:
    #         cluster_sd += str(sd) + " "

    #     output_string = "Sources: "+str(test_data[0])+", Data Means: ["+data_means+"], Data SD: ["+data_sd+"], Pair Metric: "+pair_metric+", Cluster Means: ["+cluster_means+"], Cluster SD: ["+cluster_sd+']\n'
    #     log_file.write(output_string)

    # log_file.close()



    # log_file = open('Results\\gaussian_exp2_results.txt', 'w')

    # # Get results for experiment 2
    # files = [13, 14, 15]
    # for file in files:
    #     data = Gaussian_Data()
    #     test_data = data.read_gaussian(file)
    #     test_points = test_data[2]['point']

    #     k_means = K_Means()
    #     k_means.train(test_data[0], test_points, 50)
    #     results = k_means.get_results(test_data[2]['mean'])

    #     data_mean_list = []
    #     data_sd_list = []
    #     for key, val in test_data[1].items():
    #         data_mean_list.append(val['mean'])
    #         data_sd_list.append(val['sd'])
    #     data_means = ""
    #     for mean in data_mean_list:
    #         data_means += str(mean) + " "
    #     data_sd = ""
    #     for sd in data_sd_list:
    #         data_sd += str(sd) + " "

    #     pair_metric = str(round(results[1], 5))

    #     cluster_mean_list = []
    #     cluster_sd_list = []
    #     for key, val in results[0].items():
    #         cluster_mean_list.append(round(val['mean'], 5))
    #         cluster_sd_list.append(round(val['sd'], 5))
    #     cluster_means = ""
    #     for mean in cluster_mean_list:
    #         cluster_means += str(mean) + " "
    #     cluster_sd = ""
    #     for sd in cluster_sd_list:
    #         cluster_sd += str(sd) + " "

    #     output_string = "Sources: "+str(test_data[0])+", Data Means: ["+data_means+"], Data SD: ["+data_sd+"], Pair Metric: "+pair_metric+", Cluster Means: ["+cluster_means+"], Cluster SD: ["+cluster_sd+']\n'
    #     log_file.write(output_string)
    
    # log_file.close()



    # log_file = open('Results\\gaussian_exp3_results.txt', 'w')

    # # Get results for experiment 3
    # files = [16]
    # for file in files:
    #     data = Gaussian_Data()
    #     test_data = data.read_gaussian(file)
    #     test_points = test_data[2]['point']

    #     k_means = K_Means()
    #     k_means.train(test_data[0], test_points, 50)
    #     results = k_means.get_results(test_data[2]['mean'])

    #     data_mean_list = []
    #     data_sd_list = []
    #     for key, val in test_data[1].items():
    #         data_mean_list.append(val['mean'])
    #         data_sd_list.append(val['sd'])
    #     data_means = ""
    #     for mean in data_mean_list:
    #         data_means += str(mean) + " "
    #     data_sd = ""
    #     for sd in data_sd_list:
    #         data_sd += str(sd) + " "

    #     pair_metric = str(round(results[1], 5))

    #     cluster_mean_list = []
    #     cluster_sd_list = []
    #     for key, val in results[0].items():
    #         cluster_mean_list.append(round(val['mean'], 5))
    #         cluster_sd_list.append(round(val['sd'], 5))
    #     cluster_means = ""
    #     for mean in cluster_mean_list:
    #         cluster_means += str(mean) + " "
    #     cluster_sd = ""
    #     for sd in cluster_sd_list:
    #         cluster_sd += str(sd) + " "

    #     output_string = "Sources: "+str(test_data[0])+", Data Means: ["+data_means+"], Data SD: ["+data_sd+"], Pair Metric: "+pair_metric+", Cluster Means: ["+cluster_means+"], Cluster SD: ["+cluster_sd+']\n'
    #     log_file.write(output_string)
    
    # log_file.close()



    # log_file = open('Results\\gaussian_exp4_results.txt', 'w')

    # # Get results for experiment 4
    # files = [17, 18, 19]
    # for file in files:
    #     data = Gaussian_Data()
    #     test_data = data.read_gaussian(file)
    #     test_points = test_data[2]['point']

    #     k_means = K_Means()
    #     k_means.train(test_data[0], test_points, 50)
    #     results = k_means.get_results(test_data[2]['mean'])

    #     data_mean_list = []
    #     data_sd_list = []
    #     for key, val in test_data[1].items():
    #         data_mean_list.append(val['mean'])
    #         data_sd_list.append(val['sd'])
    #     data_means = ""
    #     for mean in data_mean_list:
    #         data_means += str(mean) + " "
    #     data_sd = ""
    #     for sd in data_sd_list:
    #         data_sd += str(sd) + " "

    #     pair_metric = str(round(results[1], 5))

    #     cluster_mean_list = []
    #     cluster_sd_list = []
    #     for key, val in results[0].items():
    #         cluster_mean_list.append(round(val['mean'], 5))
    #         cluster_sd_list.append(round(val['sd'], 5))
    #     cluster_means = ""
    #     for mean in cluster_mean_list:
    #         cluster_means += str(mean) + " "
    #     cluster_sd = ""
    #     for sd in cluster_sd_list:
    #         cluster_sd += str(sd) + " "

    #     output_string = "Sources: "+str(test_data[0])+", Data Means: ["+data_means+"], Data SD: ["+data_sd+"], Pair Metric: "+pair_metric+", Cluster Means: ["+cluster_means+"], Cluster SD: ["+cluster_sd+']\n'
    #     log_file.write(output_string)
    
    # log_file.close()



    # # K-Means for housing in California
    # housing_data = pd.read_csv('Data\\California_Housing\\housing.csv')
    # housing_data = housing_data.loc[:, ["median_income", "latitude", "longitude"]]

    # # Fit model
    # k_means = K_Means()
    # k_means.train(10, housing_data, 50)



    # # K-Means for image compression
    # image = io.imread('Images\\f1_car.jpg')
    # io.imshow(image)
    # io.show()

    # # Dimension of the original image
    # rows = image.shape[0]
    # cols = image.shape[1]

    # # Flatten the image
    # image = image.reshape(rows*cols, 3)

    # # Fit model
    # k_means = K_Means()
    # k_means.train(32, image, 50)

    # # Compress image and reshape to original dimension
    # compressed_image = k_means.cluster_centers_[k_means.labels_]
    # compressed_image = compressed_image.reshape(rows, cols, 3)

    # # Save and display compressed image
    # io.imsave('compressed_image_64.png', compressed_image)
    # io.imshow(compressed_image)
    # io.show()
