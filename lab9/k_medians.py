import numpy as np
import matplotlib.pyplot as plt
import random


class Point:

    def __init__(self, x, y, cluster=-1):
        self.x = x
        self.y = y
        self.cluster = cluster

    def chebyshev_distance(self, other):
        return max(abs(self.x - other.x), abs(self.y - other.y))

    def manhattan_distance(self, other):
        return abs(self.x - other.x) + abs(self.y - other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.cluster == other.cluster

    def __ne__(self, other):
        return not self.__eq__(other)


class ClusteriserKMedians:

    def __init__(self):
        self.points = []
        self.medians = []
        self.cluster_number = -1
        self.output = ''

    def add_point(self, x, y):
        self.points.append(Point(x, y))

    def common_get_data_for_cluster(self, cluster_number):
        x_cluster, y_cluster = [], []
        for point in self.points:
            if point.cluster == cluster_number:
                x_cluster.append(point.x)
                y_cluster.append(point.y)
        return x_cluster, y_cluster

    def get_data_x_y_for_cluster(self, cluster_number):
        x_cluster, y_cluster = self.common_get_data_for_cluster(cluster_number)
        return x_cluster, y_cluster

    def get_median_for_cluster(self, cluster_number):
        x_cluster, y_cluster = self.common_get_data_for_cluster(cluster_number)
        return np.median(x_cluster), np.median(y_cluster)

    def update_manhattan_distance(self):
        for point in self.points:
            distances = [point.manhattan_distance(self.medians[i])
                         for i in range(self.cluster_number)]
            point.cluster = np.argmin(distances)

    def update_chebyshev_distance(self):
        for point in self.points:
            distances = [point.chebyshev_distance(self.medians[i])
                         for i in range(self.cluster_number)]
            point.cluster = np.argmin(distances)

    def show_current_state(self, start=False):
        markers = ['o', 'v', 'D', 's']
        fig, ax = plt.subplots(figsize=(4, 4))
        if start:
            x_cluster, y_cluster = self.get_data_x_y_for_cluster(-1)
            ax.scatter(x_cluster, y_cluster, marker='o', label='points')
        else:
            for cluster in range(self.cluster_number):
                x_cluster, y_cluster = self.get_data_x_y_for_cluster(cluster)
                ax.scatter(x_cluster, y_cluster, marker=markers[cluster], label='cluster ' + str(cluster))
            median_x, median_y = get_data_x_y_of_median(self.medians)
            ax.scatter(median_x, median_y, marker="d", label='cluster centers', c='red')

        ax.grid()
        ax.legend()

    def get_clusters(self, cluster_number=2, dist='manthattan'):
        self.cluster_number = cluster_number

        cluster_index = list(range(len(self.points)))
        cluster_index = cluster_index[:cluster_number]

        for index in cluster_index:
            self.medians.append(self.points[index])

        iteration = 0
        self.show_current_state(True)
        plt.savefig('pics/report' + str(iteration) + '.png')
        while True:
            previous_medians = self.medians.copy()

            if dist == 'manthattan':
                self.update_manhattan_distance()
            else:
                self.update_chebyshev_distance()

            self.output += f'\n{iteration}\n'
            for p in self.points:
                self.output += f'{p.x} x, {p.y} y,  cluster {p.cluster}\n'

            for cluster in range(self.cluster_number):
                median_x, median_y = self.get_median_for_cluster(cluster)
                self.medians[cluster] = Point(median_x, median_y)
                self.output += f'Median of cluster {cluster}: {median_x} x, {median_y} y\n'

            if previous_medians == self.medians:
                break
            iteration += 1
            self.show_current_state()
            plt.savefig('pics/report' + str(iteration) + '.png')

        return iteration, self.output

    def clear_clusters(self):
        for point in self.points:
            point.cluster = -1
        self.medians = []
        self.cluster_number = -1
        self.output = ''


def get_data_x_y_of_median(median):
    x_cluster, y_cluster = [], []
    for point in median:
        x_cluster.append(point.x)
        y_cluster.append(point.y)
    return x_cluster, y_cluster


if __name__ == '__main__':
    model = ClusteriserKMedians()
    model.add_point(30, 220)
    model.add_point(50, 249)
    model.add_point(271, 20)
    model.add_point(1, 253)
    model.add_point(266, 97)
    model.get_clusters(2, 'chebyshev')
