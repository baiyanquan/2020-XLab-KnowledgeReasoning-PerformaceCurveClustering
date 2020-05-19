import numpy
import matplotlib.pyplot as plt
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from tslearn.clustering import silhouette_score
from tslearn.clustering import GlobalAlignmentKernelKMeans
from tslearn.clustering import KShape
import sys

from dao.data_loader import PerformanceData

import numpy as np

performance_data = PerformanceData()
performance_data.file_name_acquire()
performance_data.csv_data_read()
performance_data.imputer()
data = performance_data.get_performance_data()


class Cluster(object):

    def __init__(self):
        self.all_time_series_data = {}

    def get_series_data(self):
        for key in data.keys():
            for service in data[key]:
                for num in data[key][service]:
                    if np.isnan(num):
                        print(key + service)
                        break
                if service not in self.all_time_series_data.keys():
                    self.all_time_series_data[service] = []
                self.all_time_series_data[service].append(data[key][service])

    def cluster_kmeans(self, min_cluster, max_cluster, metric, pre_type):
        silhouette_score_dict = {}
        sse_dict = {}
        label_dict = {}
        total_dict = {}
        min_cluster = int(min_cluster)
        max_cluster = int(max_cluster)
        for service in self.all_time_series_data.keys():
            seed = 0
            numpy.random.seed(seed)
            x_train = to_time_series_dataset(self.all_time_series_data[service])

            silhouette_score_dict["time-series-k-means"] = []
            sse_dict["time-series-k-means"] = []
            label_dict["time-series-k-means"] = {}
            for i in range(min_cluster, max_cluster):
                print(service + "-cluster:" + str(i))
                km = TimeSeriesKMeans(n_clusters=i, verbose=True)
                label = km.fit_predict(x_train)
                silhouette_score_dict["time-series-k-means"].append(silhouette_score(x_train, label, metric=metric))
                sse_dict["time-series-k-means"].append(km.inertia_)
                label_dict["time-series-k-means"][i] = label
            Cluster.generate_result(min_cluster, max_cluster, silhouette_score_dict,
                                    sse_dict, label_dict, service, "k-means")
            if pre_type == 'silhouette':
                total_dict[service] = silhouette_score_dict["time-series-k-means"]
            elif pre_type == 'sse':
                total_dict[service] = sse_dict["time-series-k-means"]
        return total_dict

    def cluster_kshape(self, min_cluster, max_cluster, metric, pre_type):
        silhouette_score_dict = {}
        sse_dict = {}
        label_dict = {}
        total_dict = {}
        min_cluster = int(min_cluster)
        max_cluster = int(max_cluster)
        for service in self.all_time_series_data.keys():
            seed = 0
            numpy.random.seed(seed)
            x_train = to_time_series_dataset(self.all_time_series_data[service])

            silhouette_score_dict["time-series-k-shape"] = []
            sse_dict["time-series-k-shape"] = []
            label_dict["time-series-k-shape"] = {}
            for i in range(min_cluster, max_cluster):
                print(service + "-cluster:" + str(i))
                km = KShape(n_clusters=i, verbose=True)
                label = km.fit_predict(x_train)
                silhouette_score_dict["time-series-k-shape"].append(silhouette_score(x_train, label, metric=metric))
                sse_dict["time-series-k-shape"].append(km.inertia_)
                label_dict["time-series-k-shape"][i] = label
            Cluster.generate_result(min_cluster, max_cluster, silhouette_score_dict, sse_dict,
                                    label_dict, service, "k-shape")
            if pre_type == 'silhouette':
                total_dict[service] = silhouette_score_dict["time-series-k-shape"]
            elif pre_type == 'sse':
                total_dict[service] = sse_dict["time-series-k-shape"]
        return total_dict

    def cluster_global_alignment_kernel_kmeans(self, min_cluster, max_cluster, metric, pre_type):
        silhouette_score_dict = {}
        sse_dict = {}
        label_dict = {}
        total_dict = {}
        min_cluster = int(min_cluster)
        max_cluster = int(max_cluster)
        for service in self.all_time_series_data.keys():
            seed = 0
            numpy.random.seed(seed)
            x_train = to_time_series_dataset(self.all_time_series_data[service])

            silhouette_score_dict["time-series-global-alignment-kernel-k-means"] = []
            sse_dict["time-series-global-alignment-kernel-k-means"] = []
            label_dict["time-series-global-alignment-kernel-k-means"] = {}
            for i in range(min_cluster, max_cluster):
                print(service + "-cluster:" + str(i))
                km = GlobalAlignmentKernelKMeans(n_clusters=i, verbose=True)
                label = km.fit_predict(x_train)
                silhouette_score_dict["time-series-global-alignment-kernel-k-means"]\
                    .append(silhouette_score(x_train, label, metric=metric))
                sse_dict["time-series-global-alignment-kernel-k-means"].append(km.inertia_)
                label_dict["time-series-global-alignment-kernel-k-means"][i] = label
            Cluster.generate_result(min_cluster, max_cluster, silhouette_score_dict, sse_dict,
                                    label_dict, service, "global-alignment-kernel-k-means")
            if pre_type == 'silhouette':
                total_dict[service] = silhouette_score_dict["time-series-global-alignment-kernel-k-means"]
            elif pre_type == 'sse':
                total_dict[service] = sse_dict["time-series-global-alignment-kernel-k-means"]
        return total_dict

    @staticmethod
    def generate_result(min_cluster, max_cluster, silhouette_score_dict,
                        sse_dict, label_dict, service, cluster_type):

            s1 = str(silhouette_score_dict)
            s2 = str(sse_dict)

            service = service.replace("/", "-")

            f = open("./result/" + cluster_type + "/" + service + '_judge.txt', 'w')
            f.write(s1 + "\n")
            f.write(s2)
            f.close()

            f = open("./result/" + cluster_type + "/" + service + '_label.txt', 'w')
            f.writelines(str(label_dict))
            f.close()

            plt.figure('Comparision of Silhouette Score')
            cluster_num_list = numpy.arange(min_cluster, max_cluster, 1)
            plt.plot(cluster_num_list, silhouette_score_dict["time-series-" + cluster_type], color='blue')

            plt.ylabel('value')
            plt.xlabel('cluster num')
            plt.savefig("./result/" + cluster_type + "/" + service + "_silhouette_score_comparision.png")
            plt.close()

            plt.figure('Comparision of SSE')
            plt.plot(cluster_num_list, sse_dict["time-series-" + cluster_type], color='blue')

            plt.ylabel('value')
            plt.xlabel('cluster num')
            plt.savefig("./result/" + cluster_type + "/" + service + "_sse_comparision.png")
            plt.close()

