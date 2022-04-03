from scipy.stats import multivariate_normal
from gaussian_data import Gaussian_Data
from matplotlib.pyplot import fill
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class GMM():

    def __init__(self, k, max_iter=None):
        self.k = k
        self.max_iter = max_iter if max_iter else 5

    def initialize(self, X_train):
        self.shape = X_train.shape
        self.n = self.shape[0]
        if len(self.shape) > 1:
            self.m = self.shape[1]

        self.phi = np.full(shape=self.k, fill_value=1/self.k)
        self.weights = np.full(shape=self.shape, fill_value=1/self.k)

        random_row = np.random.randint(low=0, high=self.n, size=self.k)
        if len(self.shape) == 1:
            self.mu = [X_train[index] for index in random_row]
            self.sigma = [np.cov(X_train.T) for _ in range(self.k)]
        else:
            self.mu = [X_train[row_index,:] for row_index in random_row]
            self.sigma = [np.cov(X_train.T) for _ in range(self.k)]

    def e_step(self, X_train):
        self.weights = self.predict_probability(X_train)
        self.phi = self.weights.mean(axis=0)

    def m_step(self, X_train):
        for i in range(self.k):
            weight = self.weights[:, [i]]
            total_weight = weight.sum()
            self.mu[i] = (X_train * weight).sum(axis=0) / total_weight
            self.sigma[i] = np.cov(X_train.T, aweights=(weight/total_weight).flatten(), bias=True)

    def train(self, X_train):
        self.initialize(X_train)

        for i in range(self.max_iter):
            self.e_step(X_train)
            self.m_step(X_train)

    def predict_probability(self, X_train):
        likelihood = np.zeros((self.n, self.k))

        for i in range(self.k):
            distribution = multivariate_normal(mean=self.mu[i], cov=self.sigma[i])
            likelihood[:, i] = distribution.pdf(X_train)
        
        numerator = likelihood * self.phi
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        weights = numerator / denominator
        return weights

    def predict(self, X_train):
        weights = self.predict_probability(X_train)
        return np.argmax(weights, axis=1)

    def get_results(self, df):
        clusters = df.predictions.unique()
        cluster_stats = {}
        for idx, i in enumerate(clusters):
            data_subset = df.loc[df['predictions'] == i]
            cluster_stats.update({idx+1: {'mean': data_subset['point'].mean(), 'sd': data_subset['point'].std()}})

        correct_count = 0
        total_count = 0
        for i in clusters:
            for j in range(100):
                data_subset = df.loc[df['predictions'] == i]
                sample_pair = data_subset.sample(n=2)
                if sample_pair.iloc[0]['mean'] == sample_pair.iloc[1]['mean']:
                    correct_count += 1
                total_count += 1
        
        return (cluster_stats, correct_count/total_count)


if __name__ == "__main__":

    Data = Gaussian_Data()
    test_data = Data.read_gaussian(14)
    test_df = test_data[2]
    test_points = np.array(test_df['point']).reshape((len(test_df), 1))

    gmm = GMM(test_data[0], 50)
    gmm.train(test_points)
    preds = gmm.predict(test_points)

    test_df['predictions'] = preds.tolist()
    results = gmm.get_results(test_df)



    # log_file = open('Results\\GMM\\gaussian_exp1_results.txt', 'w')

    # # Get results for experiment 1 datasets with generators = 3
    # files = [1, 2, 3, 4]
    # sources = [2, 3, 6, 8]
    # for file in files:
    #     for source in sources:

    #         data = Gaussian_Data()
    #         test_data = data.read_gaussian(file)
    #         test_df = test_data[2]
    #         test_points = np.array(test_df['point']).reshape((len(test_df), 1))

    #         gmm = GMM(source, 50)
    #         gmm.train(test_points)
    #         preds = gmm.predict(test_points)

    #         test_df['predictions'] = preds.tolist()
    #         results = gmm.get_results(test_df)

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
    #     test_df = test_data[2]
    #     test_points = np.array(test_df['point']).reshape((len(test_df), 1))

    #     gmm = GMM(test_data[0], 50)
    #     gmm.train(test_points)
    #     preds = gmm.predict(test_points)

    #     test_df['predictions'] = preds.tolist()
    #     results = gmm.get_results(test_df)

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



    # log_file = open('Results\\GMM\\gaussian_exp2_results.txt', 'w')

    # # Get results for experiment 2
    # files = [13, 14, 15]
    # for file in files:
    #     data = Gaussian_Data()
    #     test_data = data.read_gaussian(file)
    #     test_df = test_data[2]
    #     test_points = np.array(test_df['point']).reshape((len(test_df), 1))

    #     gmm = GMM(test_data[0], 50)
    #     gmm.train(test_points)
    #     preds = gmm.predict(test_points)

    #     test_df['predictions'] = preds.tolist()
    #     results = gmm.get_results(test_df)

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



    # log_file = open('Results\\GMM\\gaussian_exp3_results.txt', 'w')

    # # Get results for experiment 3
    # files = [16]
    # for file in files:
    #     data = Gaussian_Data()
    #     test_data = data.read_gaussian(file)
    #     test_df = test_data[2]
    #     test_points = np.array(test_df['point']).reshape((len(test_df), 1))

    #     gmm = GMM(test_data[0], 50)
    #     gmm.train(test_points)
    #     preds = gmm.predict(test_points)

    #     test_df['predictions'] = preds.tolist()
    #     results = gmm.get_results(test_df)

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



    # log_file = open('Results\\GMM\\gaussian_exp4_results.txt', 'w')

    # # Get results for experiment 4
    # files = [17, 18, 19]
    # for file in files:
    #     data = Gaussian_Data()
    #     test_data = data.read_gaussian(file)
    #     test_df = test_data[2]
    #     test_points = np.array(test_df['point']).reshape((len(test_df), 1))

    #     gmm = GMM(test_data[0], 50)
    #     gmm.train(test_points)
    #     preds = gmm.predict(test_points)

    #     test_df['predictions'] = preds.tolist()
    #     results = gmm.get_results(test_df)

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



    # # Housing data
    # plt.style.use("seaborn-whitegrid")
    # plt.rc("figure", autolayout=True)
    # plt.rc(
    #     "axes",
    #     labelweight="bold",
    #     labelsize="large",
    #     titleweight="bold",
    #     titlesize=14,
    #     titlepad=10,
    # )

    # housing_data = pd.read_csv('Data\\California_Housing\\housing.csv')
    # housing_data = housing_data.loc[:, ["median_income", "latitude", "longitude"]]

    # gmm = GMM(10, 50)
    # gmm.train(housing_data)
    # housing_data['Cluster'] = gmm.predict(housing_data)
    # housing_data['Cluster'] = housing_data['Cluster'].astype("category")
    # print(housing_data.head())

    # sns.relplot(x="longitude", y="latitude", hue="Cluster", data=housing_data, height=6)
    # plt.show()
