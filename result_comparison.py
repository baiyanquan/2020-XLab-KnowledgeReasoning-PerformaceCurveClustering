from data_loader import PerformanceData
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

all_time_series_data = {}
all_time_series_data_fault_list = {}
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
        all_time_series_data_fault_list[service].append(performance_data.fault_dict[key])
        all_time_series_data_service_list.append(service)

service_set = set(all_time_series_data_service_list)

all_label = {}
for service in service_set:
    service_file_name = service.replace("/", "-")
    with open(service_file_name + '_label.txt', "r") as f:
        original_data = f.read()
        original_data = original_data.replace("array(", "").replace(")", "").replace(", dtype=int64", "").replace("dtype=int64,", "").replace("dtype=int64", "")
        label = ast.literal_eval(original_data)
        all_label[service] = label['time-series-k-means']

# for label_key in all_label['time-series-k-means']:
#     label = all_label['time-series-k-means'][label_key]
#     for i in range(0, len(label)):
#         all_time_series_data_fault_list[i]

service_label_count = {}
for service in service_set:
    label = all_label[service][5]
    service_label_count[service] = {"cpu": {}, "network": {}, "mem": {}, "disk": {}, "k8s": {}}
    for i in range(len(label)):
        if label[i] in service_label_count[service][all_time_series_data_fault_list[service][i]].keys():
            service_label_count[service][all_time_series_data_fault_list[service][i]][label[i]] += 1
        else:
            service_label_count[service][all_time_series_data_fault_list[service][i]][label[i]] = 1

s = str(service_label_count)
f = open('label_count.txt', 'w')
f.writelines(s)
f.close()

service = "service/front-end/qps(2xx)"
cluster_num = 5
color_list = ["red", "green", "blue", "yellow", "pink", "purple"]
for i in range(len(all_time_series_data[service])):
    timestamp_list = []
    for j in range(cluster_num):
        if all_label[service][cluster_num][i] == j:
            for time in range(len(all_time_series_data[service][i])):
                timestamp_list.append(time * 10)
            plt.plot(timestamp_list, all_time_series_data[service][i], color_list[j])
plt.legend()  # 显示图例

plt.xlabel('time')
plt.ylabel('value')
plt.show()

