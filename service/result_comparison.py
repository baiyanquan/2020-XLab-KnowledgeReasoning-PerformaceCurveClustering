from dao.data_loader import PerformanceData
import numpy as np
import ast
import matplotlib.pyplot as plt

performance_data = PerformanceData()
performance_data.file_name_acquire()
performance_data.csv_data_read()
performance_data.event_data_read()
performance_data.fault_correspondence()
performance_data.imputer()
data = performance_data.get_performance_data()


class ResultComparison:

    def __init__(self):

        self.all_time_series_data = {}
        self.all_time_series_data_fault_list = {}
        self.all_time_series_data_service_list = []
        self.all_label = {}

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
                if service not in self.all_time_series_data_fault_list.keys():
                    self.all_time_series_data_fault_list[service] = []
                self.all_time_series_data_fault_list[service].append(performance_data.fault_dict[key])
                self.all_time_series_data_service_list.append(service)

    def label_list(self):
        service_set = set(self.all_time_series_data_service_list)
        try:
            for service in service_set:
                service_file_name = service.replace("/", "-")
                with open("./result/k-means/" + service_file_name + '_label.txt', "r") as f:
                    original_data = f.read()
                    original_data = original_data.replace("array(", "").replace(")", "").\
                        replace(", dtype=int64", "").replace("dtype=int64,", "").replace("dtype=int64", "")
                    label = ast.literal_eval(original_data)
                    self.all_label[service] = label['time-series-k-means']

            # for label_key in all_label['time-series-k-means']:
            #     label = all_label['time-series-k-means'][label_key]
            #     for i in range(0, len(label)):
            #         all_time_series_data_fault_list[i]

            service_label_count = {}
            for service in service_set:
                label = self.all_label[service][5]
                service_label_count[service] = {"cpu": {}, "network": {}, "mem": {}, "disk": {}, "k8s": {}}
                for i in range(len(label)):
                    if label[i] in service_label_count[service][self.all_time_series_data_fault_list[service][i]].keys():
                        service_label_count[service][self.all_time_series_data_fault_list[service][i]][label[i]] += 1
                    else:
                        service_label_count[service][self.all_time_series_data_fault_list[service][i]][label[i]] = 1

            s = str(service_label_count)
            f = open('./result/label_count/label_count.txt', 'w')
            f.writelines(s)
            f.close()
        except:
            print("KMeans Label should be generated first")

    def draw_pic(self, cluster_num, _service):
        service = _service.replace('-', "/")
        cluster_num = int(cluster_num)
        color_list = ["red", "green", "blue", "yellow", "pink", "purple",
                      "orange", "grey", "black", "brown", "burlywood", "tomato", "lawngreen", "aqua", "teal"]
        try:
            for i in range(len(self.all_time_series_data[service])):
                timestamp_list = []
                for j in range(cluster_num):
                    if self.all_label[service][cluster_num][i] == j:
                        for time in range(len(self.all_time_series_data[service][i])):
                            timestamp_list.append(time * 10)
                        plt.plot(timestamp_list, self.all_time_series_data[service][i], color_list[j])
            plt.legend()  # 显示图例

            plt.xlabel('time')
            plt.ylabel('value')
            # plt.show()
            plt.savefig("./result/comparison/" + _service + "n_cluster_" + str(cluster_num) + ".png")
        except:
            print("should have label counted first")
