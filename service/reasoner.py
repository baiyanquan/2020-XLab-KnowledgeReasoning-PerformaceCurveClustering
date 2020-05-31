import ast

class Reasoner:
    @staticmethod
    def reason_by_anomaly_expression(anomaly_expression):
        result_dict = {}

        reason_file_path = "./reason_file/reason_file.txt"
        reason_base_data = ""
        with open(reason_file_path, "r") as f:
            reason_base_data = f.read()
        f.close()
        all_result = ast.literal_eval(reason_base_data)
        if anomaly_expression in all_result.keys():
            result_dict["message"] = "found"
            result_dict["result"] = all_result[anomaly_expression]
        else:
            result_dict["message"] = "Anomaly Expression Not Found"
        return result_dict
