import numpy
import matplotlib.pyplot as plt
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from tslearn.clustering import silhouette_score
from tslearn.clustering import GlobalAlignmentKernelKMeans
from tslearn.clustering import KShape
import sys

from data_loader import PerformanceData

import numpy as np


performance_data = PerformanceData()
performance_data.file_name_acquire()
performance_data.csv_data_read()
performance_data.imputer()
data = performance_data.get_performance_data()

all_time_series_data = {}
for key in data.keys():
    for service in data[key]:
        for num in data[key][service]:
            if np.isnan(num):
                print(key + service)
                break
        if service not in all_time_series_data.keys():
            all_time_series_data[service] = []
        all_time_series_data[service].append(data[key][service])

for service in all_time_series_data.keys():
    seed = 0
    numpy.random.seed(seed)
    X_train = to_time_series_dataset(all_time_series_data[service])

    min_cluster = 2
    max_cluster = 21
    silhouette_score_dict = {}
    sse_dict = {}
    label_dict = {}

    silhouette_score_dict["time-series-k-means"] = []
    sse_dict["time-series-k-means"] = []
    label_dict["time-series-k-means"] = {}
    # silhouette_score_dict["k-shape"] = []
    # silhouette_score_dict["global-alignment-kernel-k-means"] = []
    for i in range(min_cluster, max_cluster):
        print(service + "-cluster:" + str(i))
        km = TimeSeriesKMeans(n_clusters=i, verbose=True)
        label = km.fit_predict(X_train)
        silhouette_score_dict["time-series-k-means"].append(silhouette_score(X_train, label, metric="dtw"))
        sse_dict["time-series-k-means"].append(km.inertia_)
        label_dict["time-series-k-means"][i] = label

        # km = GlobalAlignmentKernelKMeans(n_clusters=i, verbose=True)
        # label = km.fit_predict(X_train)
        # silhouette_score_dict["global-alignment-kernel-k-means"].append(silhouette_score(X_train, label, metric="dtw"))

        # km = KShape(n_clusters=i, verbose=True)
        # label = km.fit_predict(X_train)
        # silhouette_score_dict["k-shape"].append(silhouette_score(X_train, label, metric="dtw"))

    s1 = str(silhouette_score_dict)
    s2 = str(sse_dict)

    service = service.replace("/", "-")

    f = open(service + '_judge.txt', 'w')
    f.write(s1 + "\n")
    f.write(s2)
    f.close()

    f = open(service + '_label.txt', 'w')
    f.writelines(str(label_dict))
    f.close()

    plt.figure('Comparision of Silhouette Score')
    cluster_num_list = numpy.arange(min_cluster, max_cluster, 1)
    plt.plot(cluster_num_list, silhouette_score_dict["time-series-k-means"], color='blue')
    # fig1.plot(cluster_num_list, silhouette_score_dict["k-shape"], color='blue')
    # fig1.plot(cluster_num_list, silhouette_score_dict["global-alignment-kernel-k-means"], color='red')

    plt.ylabel('value')
    plt.xlabel('cluster num')
    plt.savefig(service + "_silhouette_score_comparision.png")
    plt.close()

    plt.figure('Comparision of SSE')
    plt.plot(cluster_num_list, sse_dict["time-series-k-means"], color='blue')
    # fig1.plot(cluster_num_list, silhouette_score_dict["k-shape"], color='blue')
    # fig1.plot(cluster_num_list, silhouette_score_dict["global-alignment-kernel-k-means"], color='red')

    plt.ylabel('value')
    plt.xlabel('cluster num')
    plt.savefig(service + "_sse_comparision.png")
    plt.close()
