import pandas as pd
import numpy as np
import os
import ast


class PerformanceData:

    def __init__(self, base_path="./entity/"):
        self.performance_data = {}
        self.file_name = []
        self.base_path = base_path
        self.event_file_name = []
        self.event_list = []
        self.fault_dict = {}

    def file_name_acquire(self):
        folder_name_list = os.listdir(self.base_path)
        for folder_name in folder_name_list:
            csv_name_list = os.listdir(self.base_path + folder_name + '/')
            for csv_name in csv_name_list:
                if csv_name.find(".csv") >= 0:
                    self.file_name.append(self.base_path + folder_name + '/' + csv_name)
                elif csv_name.find(".txt") >= 0:
                    self.event_file_name.append(self.base_path + folder_name + '/' + csv_name)

    def csv_data_read(self):
        for csv_name in self.file_name:
            csv_file = pd.read_csv(csv_name)
            csv_data = {}
            if len(csv_file) == 0:
                continue
            else:
                list_label = csv_file.columns.values
                # timestamp_list = csv_file['timestamp'].values - csv_file['timestamp'].values[0]
                for label in list_label:
                    if label == "timestamp" or label == "datetime":
                        continue
                    else:
                        csv_data[label] = csv_file[label].values
                self.performance_data[csv_name.split('/')[-1]] = csv_data

    def event_data_read(self):
        for event_txt_name in self.event_file_name:
            with open(event_txt_name, "r") as f:
                data = f.readlines()
                for line in data:
                    event = ast.literal_eval(line)
                    self.event_list.append(event)

    def fault_correspondence(self):
        for key in self.performance_data.keys():
            date_hour = key.split('_')[0]
            for event in self.event_list:
                b = event['start_time']
                if event['start_time'].find(date_hour) >= 0:
                    self.fault_dict[key] = event['position']

    def get_file_name(self):
        return self.file_name

    def get_performance_data(self):
        return self.performance_data

    def imputer(self):
        preheat_interval = 60
        experiment_interval = 60
        for date_key in self.performance_data.keys():
            keys = list(self.performance_data[date_key].keys())

            for key in keys:
                target_performance_data = self.performance_data[date_key][key]

                if len(target_performance_data) < 170:
                    self.performance_data[date_key].pop(key)
                    continue

                nan_count = 0
                for i in range(preheat_interval + experiment_interval):
                    if np.isnan(target_performance_data[i]):
                        nan_count += 1
                if nan_count > 0.5 * (preheat_interval + experiment_interval):
                    self.performance_data[date_key].pop(key)
                    continue

                zero_count = 0
                for i in range(preheat_interval + experiment_interval):
                    if target_performance_data[i] == 0:
                        zero_count += 1
                if zero_count > 0.7 * (preheat_interval + experiment_interval):
                    self.performance_data[date_key].pop(key)
                    continue

                i = 0
                j = 0
                while j != len(target_performance_data):
                    if not np.isnan(target_performance_data[i]):
                        i = i + 1
                        j = i
                    else:
                        while j < len(target_performance_data) and np.isnan(target_performance_data[j]):
                            j = j + 1
                        if j - i <= 6:
                            if i == 0:
                                for k in range(i, j):
                                    target_performance_data[k] = target_performance_data[j]
                            elif j == len(target_performance_data):
                                for k in range(i, j):
                                    target_performance_data[k] = target_performance_data[i - 1]
                            else:
                                divide = (target_performance_data[j] - target_performance_data[i - 1]) / (j - i + 1)
                                for k in range(i, j):
                                    target_performance_data[k] = target_performance_data[i - 1] + (k - i + 1) * divide
                        else:
                            for k in range(i, j):
                                target_performance_data[k] = 0
                self.performance_data[date_key][key] = target_performance_data

        date_keys = list(self.performance_data.keys())
        for date_key in date_keys:
            if len(self.performance_data[date_key]) == 0:
                self.performance_data.pop(date_key)
