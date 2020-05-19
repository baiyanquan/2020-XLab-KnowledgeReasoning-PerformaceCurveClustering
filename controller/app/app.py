from flask import Flask, jsonify
from service.cluster import Cluster
from service.picture_construct import PictureConstruct
from service.result_comparison import ResultComparison

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


@app.route('/api/v1.0/Cluster/KShape/sse/<min_cluster>/<max_cluster>/<metric>', methods=['POST'])
def get_sse_kshape(min_cluster, max_cluster, metric):
    return jsonify(cluster.cluster_kshape(min_cluster, max_cluster, metric, "sse"))


@app.route('/api/v1.0/Cluster/KShape/silhouette/<min_cluster>/<max_cluster>/<metric>', methods=['POST'])
def get_silhouette_kshape(min_cluster, max_cluster, metric):
    return jsonify(cluster.cluster_kshape(min_cluster, max_cluster, metric, "silhouette"))


@app.route('/api/v1.0/Cluster/GlobalAlignmentKernelKMeans/sse/<min_cluster>/<max_cluster>/<metric>', methods=['POST'])
def get_sse_g_kmeans(min_cluster, max_cluster, metric):
    return jsonify(cluster.cluster_global_alignment_kernel_kmeans(min_cluster, max_cluster, metric, "sse"))


@app.route('/api/v1.0/Cluster/GlobalAlignmentKernelKMeans/silhouette/<min_cluster>/<max_cluster>/<metric>', methods=['POST'])
def get_silhouette_g_kmeans(min_cluster, max_cluster, metric):
    return jsonify(cluster.cluster_global_alignment_kernel_kmeans(min_cluster, max_cluster, metric, "silhouette"))


app.route('/api/v1.0/')
if __name__ == '__main__':
    app.run()
