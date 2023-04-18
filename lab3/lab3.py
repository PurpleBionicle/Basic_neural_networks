import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.tan(x)


def make_plot(point_for_training, x_for_training, point_for_testing, prediction, norma, epochs, window_size,
              error_list, id: int = 0) -> None:
    """
    Строит два графика - один кол-во ошибок в каждую эпоху, второй график предсказанных значений и реальных
    """
    plt.plot(point_for_training, x_for_training, '-ob', point_for_testing, f(point_for_testing), '-or',
             point_for_testing, prediction, 'og')
    plt.legend(['train', 'test', 'predicted'])
    plt.xlabel(f'norm_edu = {norma}, M = {epochs}, p = {window_size}')
    plt.savefig(f'result_{id}.png')
    plt.show()
    plt.plot(error_list)
    plt.xlabel('epoch number')
    plt.ylabel('error')
    plt.title('Error(epoch)')
    plt.savefig(f'error_{id}.png')
    plt.show()


def network_for_points(norma: float, epochs: int, window_size: int, id: int):
    """
    НС для предсказания значений функций по предыдущим значеним
    :param norma: норма обучения
    :param epochs: кол-во эпох
    :param window_size: размер окна для обучения
    :return: None
    """
    a, b, count_of_points = 2, 3, 20
    weights: list[float] = np.array([0.] * (window_size + 1))
    point_for_training = np.linspace(a, b, count_of_points,
                                     endpoint=True)  # равномерно распределенные точки в исходном отрезке
    point_for_testing = np.linspace(b, 2 * b - a, count_of_points,
                                    endpoint=True)  # равномерно распределенные точки на отрезке с будущими предсказанными точками
    x_for_training = f(point_for_training)
    all_windows = np.array(
        [[x_for_training[i] for i in range(x, x + window_size)] + [1] for x in range(count_of_points - window_size)])
    x_right = x_for_training[window_size:]
    error_list = []
    """Обучение на исходном отрезке"""
    for epoch in range(epochs):
        sqr_delta = 0
        for i, training_x in enumerate(all_windows):
            net = np.dot(training_x, weights)
            error = x_right[i] - net
            sqr_delta += error ** 2
            delta = np.dot(np.array([training_x]).T, np.array([norma * error]))
            weights += delta
        error_list.append(sqr_delta)
    prediction = []
    x_prediction = x_for_training[-window_size:]
    """Предсказание"""
    for i in range(count_of_points):
        x_prediction = np.append(x_prediction, [1])
        net = np.dot(x_prediction, weights)
        prediction.append(net)
        x_prediction = np.append(x_prediction[1:-1], [net])

    make_plot(point_for_training, x_for_training, point_for_testing, prediction, norma, epochs, window_size,
              error_list, id=id)
    print(weights)


def main():
    network_for_points(0.3, 1000, 2, id=1)
    network_for_points(0.3, 2000, 2, id=2)
    network_for_points(0.3, 10000, 3, id=3)
    network_for_points(0.1, 10000, 3, id=4)


if __name__ == '__main__':
    main()
