import os

from flask import Flask, jsonify, request
from service.cluster import Cluster
from service.picture_construct import PictureConstruct
from service.result_comparison import ResultComparison
from flask_cors import CORS


cluster = Cluster()
cluster.get_series_data()

pictureConstruct = PictureConstruct()

resultComparison = ResultComparison()
resultComparison.get_series_data()

app = Flask(__name__)


@app.route('/api/v1.0/Cluster/KMeans/sse/<min_cluster>/<max_cluster>/<metric>', methods=['POST'])
def get_sse_kmeans(min_cluster, max_cluster, metric):
    return jsonify(cluster.cluster_kmeans(min_cluster, max_cluster, metric, "sse"))


@app.route('/api/v1.0/Cluster/KMeans/silhouette/<min_cluster>/<max_cluster>/<metric>', methods=['POST'])
def get_silhouette_kmeans(min_cluster, max_cluster, metric):
    return jsonify(cluster.cluster_kmeans(min_cluster, max_cluster, metric, "silhouette"))


@app.route('/api/v1.0/Pic/KMeans/<service>', methods=['POST'])
def get_pic_kmeans(service):
    pictureConstruct.generate_pic(service)
    return "picture constructed"


@app.route('/api/v1.0/LableCount/Kmeans', methods=['POST'])
def get_lable_count():
    resultComparison.label_list()
    return "label_count generated"


@app.route('/api/v1.0/Comparison/KMeans/<service>/<cluster_num>', methods=['POST'])
def get_comparison_kmeans(service, cluster_num):
    resultComparison.draw_pic(cluster_num, service)
    return "result generated"


@app.route('/api/v1.0/cluster_test', methods=['POST'])
def cluster_test():
    model = request.form['model']
    distance_type = request.form['distance_type']
    folder = request.form['folder']

    return jsonify(cluster.cluster_test(model, distance_type, folder))


@app.route('/api/v1.0/analyse_test', methods=['POST'])
def analyse_test():
    model = request.form['model']
    distance_type = request.form['distance_type']
    folder = request.form['folder']

    return jsonify(cluster.analyse_test(model, distance_type, folder))


@app.route('/api/v1.0/cluster_center_test', methods=['POST'])
def cluster_center_test():
    model = request.form['model']
    distance_type = request.form['distance_type']
    folder = request.form['folder']

    return jsonify(cluster.cluster_center_test(model, distance_type, folder))


@app.route('/api/v1.0/cluster_center', methods=['POST'])
def cluster_center():
    service = request.form['service']
    model = request.form['model']
    distance_type = request.form['distance_type']
    cluster_num = request.form['cluster_num']
    folder = request.form['folder']

    return jsonify(cluster.acquire_cluster_center(service, model, distance_type, cluster_num, folder))


@app.route('/api/v1.0/default_cluster', methods=['POST'])
def default_cluster():
    model = request.form['model']
    distance_type = request.form['distance_type']
    folder = request.form['folder']

    return jsonify(cluster.default_all_cluster(model, distance_type, folder))


@app.route('/api/v1.0/default_analyse', methods=['POST'])
def default_analyse():
    model = request.form['model']
    distance_type = request.form['distance_type']
    folder = request.form['folder']

    return jsonify(cluster.default_all_analyse(model, distance_type, folder))


@app.route('/api/v1.0/cluster', methods=['POST'])
def get_cluster_result():
    service = request.form['service']
    model = request.form['model']
    distance_type = request.form['distance_type']
    cluster_num = request.form['cluster_num']
    folder = request.form['folder']

    # file name: service - model - distance_type - cluster_num

    return jsonify(cluster.cluster(service, model, distance_type, cluster_num, folder))


@app.route('/api/v1.0/generate_knowledge', methods=['POST'])
def generate_knowledge():
    model = request.form['model']
    distance_type = request.form['distance_type']
    folder = request.form['folder']

    return jsonify(cluster.generate_knowledge(model, distance_type, folder))


@app.route('/api/v1.0/predict', methods=['POST'])
def predict():
    service = request.form['service']
    model = request.form['model']
    distance_type = request.form['distance_type']
    folder = request.form['folder']
    file = request.files['file']

    return jsonify(cluster.predict(service, model, distance_type, folder, file))


app.route('/api/v1.0/')
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
if __name__ == '__main__':
    app.run()
