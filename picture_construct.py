import matplotlib.pyplot as plt
from data_loader import PerformanceData

performance_data = PerformanceData()
performance_data.file_name_acquire()
performance_data.csv_data_read()
performance_data.event_data_read()
performance_data.fault_correspondence()
performance_data.imputer()
data = performance_data.get_performance_data()

plt.title('Performance Data Line')
num = 0
fault_type = {"cpu": "yellow", "network": "green", "mem": "blue", "disk": "purple", "k8s": "red"}

for key in data.keys():
    for service in data[key]:
        # timestamp_list = []
        # for time in range(len(data[key][service])):
        #     timestamp_list.append(time * 10)
        # num = num + 1
        # plt.plot(timestamp_list, data[key][service])
        if service == 'service/front-end/qps(2xx)':
            timestamp_list = []
            for time in range(len(data[key][service])):
                timestamp_list.append(time * 10)
            num = num + 1
            plt.plot(timestamp_list, data[key][service], fault_type[performance_data.fault_dict[key]])
print(num)
plt.legend()  # 显示图例

plt.xlabel('time')
plt.ylabel('value')
plt.show()
