from math import sqrt
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style
from collections import Counter
import warnings


def init():
    style.use('fivethirtyeight')


def find_k_nearest_neighbours(predict_set, data_set, k=3):
    if len(data_set) >= k:
        warnings.warn(f'total_voting_groups={len(data_set)} is greater than or equal to k={k}')

    distances_pair = {}
    for predict_data in predict_set:
        distances = []

        for group, features in data_set.items():
            euclidean_distance = np.linalg.norm(np.array(predict_data) - np.array(features))
            distances.append([euclidean_distance, group])

        predict_data_key = ','.join([str(x) for x in predict_data])
        distances_pair[predict_data_key] = {'data': predict_data, 'distances': distances}

    vote_results = {}
    for key, distance_data in distances_pair.items():
        votes = [i[1] for i in sorted(distance_data['distances'])[:k]]
        vote_result = Counter(votes).most_common(1)[0][0]
        vote_results[key] = {'data': distance_data['data'], 'class': vote_result}

    print(vote_results)
    return vote_results


def graph(classes):
    for k, features in classes.items():
        [plt.scatter(x[0], x[1], s=15, color=k) for x in features]


def graph_k_nearest_neighbours(data_set):
    [plt.scatter(data['data'][0], data['data'][1], s=30, color=data['class']) for data in data_set.values()]


def run():
    classes = {
        'k': [[1, 2], [2, 3], [3, 1]],
        'r': [[6, 5], [7, 7], [8, 6]],
    }
    new_features = [[2, 4], [5, 7]]

    k_nearest_neighbours = find_k_nearest_neighbours(predict_set=new_features, data_set=classes, k=3)

    graph(classes)
    graph_k_nearest_neighbours(k_nearest_neighbours)
    plt.show()


def main():
    init()
    run()


if __name__ == '__main__':
    main()

