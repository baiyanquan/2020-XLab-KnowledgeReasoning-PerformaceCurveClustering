import numpy
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from tslearn.clustering import silhouette_score
from tslearn.clustering import GlobalAlignmentKernelKMeans
from tslearn.clustering import KShape
import csv
import sys
import os
import ast
import pandas as pd

from dao.data_loader import PerformanceData

import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

performance_data = PerformanceData()
performance_data.file_name_acquire()
performance_data.csv_data_read()
performance_data.event_data_read()
performance_data.fault_correspondence()
performance_data.imputer()
data = performance_data.get_performance_data()

all_time_series_data = {}
all_time_series_data_fault_list = {}
all_time_series_data_fault_detail_list = {}
all_time_series_data_service_list = []
for key in data.keys():
    for service in data[key]:
        for num in data[key][service]:
            if np.isnan(num):
                print(key + service)
                break
        if service not in all_time_series_data.keys():
            all_time_series_data[service] = []
        all_time_series_data[service].append(data[key][service])
        if service not in all_time_series_data_fault_list.keys():
            all_time_series_data_fault_list[service] = []
            all_time_series_data_fault_detail_list[service] = []
        all_time_series_data_fault_list[service].append(performance_data.fault_dict[key])
        all_time_series_data_fault_detail_list[service].append(performance_data.fault_detail_dict[key])
        all_time_series_data_service_list.append(service)

service_set = set(all_time_series_data_service_list)


class Cluster(object):

    def __init__(self):
        self.all_time_series_data = {}
        self.default_cluster_num = {
            "TimeSeriesKMeans-softdtw": {
                "service/carts/latency": 4,
                "service/carts/qps(2xx)": 5,
                "service/catalogue/latency": 3,
                "service/catalogue/qps(2xx)": 5,
                "service/front-end/latency": 5,
                "service/front-end/qps(2xx)": 3,
                "service/orders/latency": 3,
                "service/orders/qps(2xx)": 5,
                "service/payment/latency": 4,
                "service/payment/qps(2xx)": 4,
                "service/shipping/latency": 6,
                "service/shipping/qps(2xx)": 4,
                "service/user/latency": 4,
                "service/user/qps(2xx)": 4,
            },
            "TimeSeriesKMeans-dtw": {
                "service/carts/latency": 5,
                "service/carts/qps(2xx)": 5,
                "service/catalogue/latency": 5,
                "service/catalogue/qps(2xx)": 5,
                "service/front-end/latency": 4,
                "service/front-end/qps(2xx)": 7,
                "service/orders/latency": 5,
                "service/orders/qps(2xx)": 7,
                "service/payment/latency": 5,
                "service/payment/qps(2xx)": 5,
                "service/shipping/latency": 4,
                "service/shipping/qps(2xx)": 6,
                "service/user/latency": 4,
                "service/user/qps(2xx)": 5,
            },
            "KShape-softdtw": {
                "service/carts/latency": 4,
                "service/carts/qps(2xx)": 5,
                "service/catalogue/latency": 3,
                "service/catalogue/qps(2xx)": 5,
                "service/front-end/latency": 5,
                "service/front-end/qps(2xx)": 3,
                "service/orders/latency": 3,
                "service/orders/qps(2xx)": 5,
                "service/payment/latency": 4,
                "service/payment/qps(2xx)": 4,
                "service/shipping/latency": 6,
                "service/shipping/qps(2xx)": 4,
                "service/user/latency": 4,
                "service/user/qps(2xx)": 4,
            },
            "KShape-dtw": {
                "service/carts/latency": 4,
                "service/carts/qps(2xx)": 5,
                "service/catalogue/latency": 3,
                "service/catalogue/qps(2xx)": 5,
                "service/front-end/latency": 5,
                "service/front-end/qps(2xx)": 3,
                "service/orders/latency": 3,
                "service/orders/qps(2xx)": 5,
                "service/payment/latency": 4,
                "service/payment/qps(2xx)": 4,
                "service/shipping/latency": 6,
                "service/shipping/qps(2xx)": 4,
                "service/user/latency": 4,
                "service/user/qps(2xx)": 4,
            },
        }
        self.color_list = ["red", "green", "blue", "yellow", "pink", "purple", "aliceblue", "brown"]

    def imputer(self, test_data):
        preheat_interval = 60
        experiment_interval = 60
        if len(test_data) < 170:
            return []
        nan_count = 0
        for i in range(preheat_interval + experiment_interval):
            if np.isnan(test_data[i]):
                nan_count += 1
        if nan_count > 0.5 * (preheat_interval + experiment_interval):
            return []

        zero_count = 0
        for i in range(preheat_interval + experiment_interval):
            if test_data[i] == 0:
                zero_count += 1
        if zero_count > 0.7 * (preheat_interval + experiment_interval):
            return []

        i = 0
        j = 0
        while j != len(test_data):
            if not np.isnan(test_data[i]):
                i = i + 1
                j = i
            else:
                while j < len(test_data) and np.isnan(test_data[j]):
                    j = j + 1
                if j - i <= 6:
                    if i == 0:
                        for k in range(i, j):
                            test_data[k] = test_data[j]
                    elif j == len(test_data):
                        for k in range(i, j):
                            test_data[k] = test_data[i - 1]
                    else:
                        divide = (test_data[j] - test_data[i - 1]) / (j - i + 1)
                        for k in range(i, j):
                            test_data[k] = test_data[i - 1] + (k - i + 1) * divide
                else:
                    for k in range(i, j):
                        test_data[k] = 0
        return test_data

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

    def cluster_test(self, model, distance_type, folder):
        result_dict = {}
        latency_min_cluster = 3
        latency_max_cluster = 8
        qps_min_cluster = 4
        qps_max_cluster = 9
        for service in self.all_time_series_data.keys():
            min_cluster = 1
            max_cluster = 10
            if "latency" in service:
                min_cluster = latency_min_cluster
                max_cluster = latency_max_cluster
            else:
                min_cluster = qps_min_cluster
                max_cluster = qps_max_cluster
            for i in range(min_cluster, max_cluster):
                result_dict[service + str(i)] = self.cluster(service, model, distance_type, i, folder)
        return result_dict

    def analyse_test(self, model, distance_type, folder):
        result_dict = {}
        latency_min_cluster = 3
        latency_max_cluster = 8
        qps_min_cluster = 4
        qps_max_cluster = 9
        for service in self.all_time_series_data.keys():
            min_cluster = 1
            max_cluster = 10
            if "latency" in service:
                min_cluster = latency_min_cluster
                max_cluster = latency_max_cluster
            else:
                min_cluster = qps_min_cluster
                max_cluster = qps_max_cluster
            for i in range(min_cluster, max_cluster):
                result_dict[service + str(i)] = self.analyse(service, model, distance_type, i, folder)
        return result_dict

    def cluster_center_test(self, model, distance_type, folder):
        result_dict = {}
        latency_min_cluster = 3
        latency_max_cluster = 8
        qps_min_cluster = 4
        qps_max_cluster = 9
        for service in self.all_time_series_data.keys():
            min_cluster = 1
            max_cluster = 10
            if "latency" in service:
                min_cluster = latency_min_cluster
                max_cluster = latency_max_cluster
            else:
                min_cluster = qps_min_cluster
                max_cluster = qps_max_cluster
            for i in range(min_cluster, max_cluster):
                result_dict[service + str(i)] = self.acquire_cluster_center(service, model, distance_type, i, folder)
        return result_dict

    def default_all_cluster(self, model, distance_type, folder):
        result_dict = {}
        default_cluster_num = self.default_cluster_num[model + '-' + distance_type]
        for i in default_cluster_num.keys():
            result_dict[i] = self.cluster(i, model, distance_type, default_cluster_num[i], folder)
        return result_dict

    def default_all_analyse(self, model, distance_type, folder):
        result_dict = {}
        default_cluster_num = self.default_cluster_num[model + '-' + distance_type]
        for i in default_cluster_num.keys():
            result_dict[i] = self.analyse(i, model, distance_type, default_cluster_num[i], folder)
        return result_dict

    def cluster(self, service, model, distance_type, cluster_num, folder):
        if not os.path.exists('./result/' + folder):
            os.makedirs('./result/' + folder)
            os.makedirs('./result/' + folder + '/model')
            os.makedirs('./result/' + folder + '/judge')

        service_file_name = service.replace("/", "-")
        result_dict = {}

        cluster_num = int(cluster_num)

        random_state = 0
        x_train = to_time_series_dataset(self.all_time_series_data[service])
        km = TimeSeriesKMeans(n_clusters=cluster_num, metric=distance_type, verbose=True)
        if model == 'TimeSeriesKMeans':
            pass
        elif model == 'GlobalAlignmentKernelKMeans':
            km = GlobalAlignmentKernelKMeans(n_clusters=cluster_num, verbose=True)
        elif model == 'KShape':
            km = KShape(n_clusters=cluster_num, verbose=True)
        else:
            result_dict["message"] = "Incorrect model type"
            return result_dict

        label = km.fit_predict(x_train)
        model_path = './result/' + folder + '/model' + '/' + service_file_name + '-' + model + '-' + distance_type + \
                     '-' + str(cluster_num) + '.h5py'
        if os.path.exists(model_path):
            os.remove(model_path)
        km.to_hdf5(model_path)

        sc = silhouette_score(x_train, label, metric=distance_type)
        sse = km.inertia_
        result_dict['label'] = str(label)
        result_dict['sc'] = sc
        result_dict['sse'] = sse

        f = open('./result/' + folder + '/judge' + '/' + service_file_name + '-' + model + '-' + distance_type + '-' +
                 str(cluster_num) + '_label.txt', 'w')
        f.write(str(result_dict))
        f.close()

        return result_dict

    def predict(self, service, model, distance_type, folder, file):
        result_dict = {}

        csv_file = pd.read_csv(file)
        test_data = []
        check_service = ""
        if len(csv_file) == 0:
            result_dict["message"] = "Invalid file"
            return result_dict
        else:
            if len(csv_file.columns.values) < 1:
                result_dict["message"] = "Invalid file"
                return result_dict
            label = csv_file.columns.values[0]
            if label.split("/")[1] != service:
                result_dict["message"] = "Please check file and chosen service"
                return result_dict
            else:
                test_data = self.imputer(csv_file[label].values)
                if len(test_data) == 0:
                    result_dict["message"] = "Invalid data"
                    return result_dict
                check_service = label
        default_cluster_num = self.default_cluster_num[model + '-' + distance_type][check_service]

        check_service_file = check_service.replace("/", "-")

        km = TimeSeriesKMeans(n_clusters=default_cluster_num, metric=distance_type, verbose=True)
        model_path = './result/' + folder + '/model' + '/' + check_service_file + '-' + model + '-' + distance_type + '-'\
                     + str(default_cluster_num) + '.h5py'
        if os.path.exists(model_path):
            km = TimeSeriesKMeans.from_hdf5(model_path)
            if model == 'TimeSeriesKMeans':
                pass
            elif model == 'GlobalAlignmentKernelKMeans':
                km = GlobalAlignmentKernelKMeans.from_hdf5(model_path)
            elif model == 'KShape':
                km = KShape.from_hdf5(model_path)
            else:
                result_dict["message"] = "Incorrect model type"
                return result_dict

        else:
            result_dict["message"] = "Please cluster first"
            return result_dict

        result_dict["result"] = check_service.replace("service/", "").replace("/", "-") + ':' + str(km.predict(to_time_series_dataset(test_data))[0])

        return result_dict

    def analyse(self, service, model, distance_type, cluster_num, folder):
        result_dict = {}
        service_file_name = service.replace("/", "-")

        label_path = './result/' + folder + '/judge' + '/' + service_file_name + '-' + model + '-' + distance_type + \
                     '-' + str(cluster_num) + '_label.txt'
        if os.path.exists(label_path):
            label = []
            with open(label_path, "r") as f:
                original_data = f.read()
                original_data = original_data.replace(": inf", ": 'inf'")
                label = list(map(int, ast.literal_eval(original_data)['label'].replace("[", "").replace("]", "")
                                 .split(" ")))
            f.close()
            service_performance_data = self.all_time_series_data[service]
            plt.figure(service + '-' + model + '-' + distance_type + '-' + str(cluster_num)
                       + '_cluster_result.png')
            for i in range(len(service_performance_data)):
                timestamp_list = []
                for j in range(cluster_num):
                    if label[i] == j:
                        for time in range(len(service_performance_data[i])):
                            timestamp_list.append(time * 10)
                        plt.plot(timestamp_list, service_performance_data[i], self.color_list[j])
            plt.legend()  # 显示图例

            plt.xlabel('time')
            plt.ylabel('value')

            figure_path = './result/' + folder + '/figure' + '/' + service_file_name + '/' + model + '-' + \
                          distance_type + '-' + str(cluster_num) + '_cluster_result.png'
            if not os.path.exists('./result/' + folder + '/figure'):
                os.makedirs('./result/' + folder + '/figure')
            if not os.path.exists('./result/' + folder + '/figure' + '/' + service_file_name):
                os.makedirs('./result/' + folder + '/figure' + '/' + service_file_name)
            plt.savefig(figure_path)
            plt.close()

        else:
            result_dict["message"] = "Please cluster first"
            return result_dict

        result_dict["message"] = "Success"
        return result_dict

    def acquire_cluster_center(self, service, model, distance_type, cluster_num, folder):
        result_dict = {}
        service_file_name = service.replace("/", "-")
        cluster_num = int(cluster_num)

        km = TimeSeriesKMeans(n_clusters=cluster_num, metric=distance_type, verbose=True)
        model_path = './result/' + folder + '/model' + '/' + service_file_name + '-' + model + '-' + distance_type + '-' \
                     + str(cluster_num) + '.h5py'
        if os.path.exists(model_path):
            km = TimeSeriesKMeans.from_hdf5(model_path)
            if model == 'TimeSeriesKMeans':
                pass
            elif model == 'GlobalAlignmentKernelKMeans':
                km = GlobalAlignmentKernelKMeans.from_hdf5(model_path)
            elif model == 'KShape':
                km = KShape.from_hdf5(model_path)
            else:
                result_dict["message"] = "Incorrect model type"
                return result_dict

        else:
            result_dict["message"] = "Please cluster first"
            return result_dict

        cluster_center = km.cluster_centers_

        plt.figure(service + '-' + model + '-' + distance_type + '-' + str(cluster_num)
                   + '_cluster_center.png')
        for i in range(cluster_num):
            timestamp_list = []
            for time in range(len(cluster_center[i])):
                timestamp_list.append(time * 10)
            plt.plot(timestamp_list, cluster_center[i], self.color_list[i])
        plt.legend()  # 显示图例

        plt.xlabel('time')
        plt.ylabel('value')

        figure_path = './result/' + folder + '/cluster_center_figure' + '/' + service_file_name + '/' + model + '-' + \
                      distance_type + '-' + str(cluster_num) + '_cluster_center.png'
        if not os.path.exists('./result/' + folder + '/cluster_center_figure'):
            os.makedirs('./result/' + folder + '/cluster_center_figure')
        if not os.path.exists('./result/' + folder + '/cluster_center_figure' + '/' + service_file_name):
            os.makedirs('./result/' + folder + '/cluster_center_figure' + '/' + service_file_name)
        plt.savefig(figure_path)
        plt.close()

        result_dict["message"] = "Success"

        return result_dict

    def generate_knowledge(self, model, distance_type, folder):
        result_dict = {}
        default_cluster_num = self.default_cluster_num[model + '-' + distance_type]

        knowledge_list = []
        service_label_count = {}
        for service in default_cluster_num.keys():
            service_file_name = service.replace("/", "-")
            cluster_num = default_cluster_num[service]
            service_label_count[service] = {}

            label_path = './result/' + folder + '/judge' + '/' + service_file_name + '-' + model + '-' + distance_type \
                         + '-' + str(cluster_num) + '_label.txt'
            if os.path.exists(label_path):
                label = []
                with open(label_path, "r") as f:
                    original_data = f.read()
                    original_data = original_data.replace(": inf", ": 'inf'")
                    label = list(map(int, ast.literal_eval(original_data)['label'].replace("[", "").replace("]", "")
                                     .split(" ")))
                f.close()

                service_performance_data = self.all_time_series_data[service]

                for i in range(len(label)):
                    a = all_time_series_data_fault_list[service]
                    if label[i] in service_label_count[service].keys():
                        service_label_count[service][label[i]][all_time_series_data_fault_list[service][i]] += 1
                    else:
                        service_label_count[service][label[i]] = {"cpu": 0, "network": 0, "mem": 0, "disk": 0, "k8s": 0}
                        service_label_count[service][label[i]][all_time_series_data_fault_list[service][i]] += 1

                    knowledge = []
                    knowledge.append(all_time_series_data_fault_detail_list[service][i]["position"])
                    if knowledge[0] == "k8s":
                        target_pod_info = all_time_series_data_fault_detail_list[service][i]["cmd"].split(" ")[
                            -1].split("-")
                        target_pod = target_pod_info[0]
                        for l in range(1, len(target_pod_info) - 2):
                            target_pod += "-"
                            target_pod += target_pod_info[l]
                        knowledge.append("pod:" + target_pod)
                    else:
                        knowledge.append("server:" + all_time_series_data_fault_detail_list[service][i]["ip"])
                    service_info = service.split("/")
                    knowledge.append("service:" + service_info[-2])
                    knowledge.append(service_info[-2] + "-" + service_info[-1] + ":" + str(label[i]))
                    knowledge_list.append(knowledge)

            else:
                result_dict["message"] = "Please cluster first"
                return result_dict

        f = open('./result/' + folder + 'knowledge_info.txt', 'w')
        for i in knowledge_list:
            for j in range(len(i) - 1):
                f.write(str(i[j]) + '\t')
            f.write(str(i[-1] + '\n'))
        f.close()

        f = open('./result/' + folder + 'performance_statistics.txt', 'w')
        f.write(str(service_label_count))
        f.close()

        f = open('./result/' + folder + 'performance_statistics.csv', 'w', newline='')
        csv_writer = csv.writer(f)
        csv_writer.writerow(["", "type", "cpu", "network", "mem", "disk", "k8s"])
        for service in service_label_count:
            service_type = service_label_count[service]
            for i in range(len(service_type.keys())):
                csv_writer.writerow([service, str(i), service_type[i]["cpu"], service_type[i]["network"], service_type[i]["mem"], service_type[i]["disk"], service_type[i]["k8s"]])
        f.close()

        result_dict["message"] = "Success"
        return result_dict


