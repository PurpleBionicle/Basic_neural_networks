from math import exp
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

from prettytable import PrettyTable


def make_plot(errors: list[int], is_function_binary: bool, is_least_combo=False) -> None:
    """
    рисуем график
    :param errors: ошибки за каждую эпоху
    """
    file_name: str = 'binary' if is_function_binary else 'softsign'
    file_name += '_least_combo' if is_least_combo else ''
    plt.plot([i for i in range(len(errors))], errors, label=f"{file_name} function Error(epoch)")
    plt.legend()
    plt.xlabel('epoch numbers')
    plt.ylabel('error')
    plt.grid(True)
    plt.savefig(f'{file_name}.png')
    plt.show()


def boolean_function(x: list[int]) -> int:
    """
    :param x: набор значений переменных
    :return: результат функции
    """
    return int((x[0] or x[1]) and x[2] and x[3])


def loss(y_true, y_predicted):
    return y_true - y_predicted


def fast_get_all_predictions(network, x_test):
    return np.array([network.working_mode(x) for x in x_test])


class RadialBasisFunctionNeuron:
    def __init__(self, center):
        self.center = center

    def get_phi(self, x_values):
        summ = sum(pow(x_values - self.center, 2))
        return exp(-summ)


class Net:
    def __init__(self, is_function_binary: bool, neurons_count: int, centers, weights=None):
        self.is_function_binary = is_function_binary
        self.neurons_count = neurons_count
        self.neuron = [RadialBasisFunctionNeuron(centers[i]) for i in range(self.neurons_count)]
        self.W = [0] * neurons_count if weights == None else weights
        self.bias = 0  # offset

    def working_mode(self, x, threshold=0.5):
        """
        Ф-ция активации
        """
        self.phi = [self.neuron[i].get_phi(x) for i in range(self.neurons_count)]
        self.net = np.dot(self.W, self.phi) + self.bias
        if self.is_function_binary:
            self.binary_step = int(self.net >= 0)
            return self.binary_step
        else:
            self.softsign = 1 / (1 + exp(-self.net))
            return int(self.softsign > threshold)

    def studying_mode(self, delta, norma=0.3):
        """
            Корректировка весов синопсов
        """
        if self.is_function_binary:
            self.W += np.dot(norma * delta, self.phi)  # +dW
            self.bias += norma * delta
        else:
            self.dz = 0.5 / (1 + abs(self.softsign))
            self.dW = np.dot(norma * delta * self.dz, self.phi)
            self.db = norma * delta * self.dz
            self.W += self.dW
            self.bias += self.db


def initial_function() -> tuple[list[list[int] | Any], list[int], Any]:
    """
    :return: наборы значений переменных и значение функции
    """
    x = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
         [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
         [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
         [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]]
    c = np.array([[0, 1, 1, 1],
                  [1, 0, 1, 1],
                  [1, 1, 1, 1]])
    return x, [boolean_function(x[i]) for i in range(len(x))], c


def bypass(is_function_binary: bool, norma: float):
    mytable1 = PrettyTable().copy()
    mytable1.field_names = ["Epoch number", "Вектор весов w", "Выходной вектор y", "Суммарная ошибка Е"]
    x_test, y, centers = initial_function()

    network = Net(is_function_binary, 3, centers)
    errors_per_epoch = []
    count_of_test_values: int = len(x_test)
    error: int = len(x_test)
    epoch: int = 0
    while error > 0:
        error = 0
        predicted_values: list = []
        for i in range(count_of_test_values):
            x = x_test[i]
            y_true = y[i]
            predicted = network.working_mode(x)
            delta = loss(y_true, predicted)
            error += np.abs(delta)
            network.studying_mode(delta, norma)
            predicted_values.append(predicted)
        errors_per_epoch.append(error)
        epoch += 1

        mytable1.add_row([epoch, network.W + [network.bias], predicted_values, error])
    mytable1.float_format = '.3'
    print(f'{mytable1}\n')
    make_plot(errors_per_epoch, is_function_binary)


def find_least_combination(is_function_binary: bool):
    """
        нахождение минимального обущающего набора векторов
        """
    print('Combinations for binary FA' if is_function_binary else 'Combinations for softsign FA')
    mytable1 = PrettyTable().copy()
    mytable1.field_names = ["length of vectors", "combination of vectors"]
    for number_of_variables in range(16 - 1, 0, -1):
        combs = list(combinations(range(16), number_of_variables))
        x_test, y, c = initial_function()
        limit_epoch_for_solution: int = 200
        for indexes in combs:
            net = Net(is_function_binary, 3, c)
            predicted_values = []
            for epoch in range(limit_epoch_for_solution):
                error_in_current_epoch: int = 0
                for i in indexes:
                    x = x_test[i]
                    y_true = y[i]
                    y_predicted = net.working_mode(x)
                    delta = loss(y_true, y_predicted)
                    error_in_current_epoch += abs(delta)
                    net.studying_mode(delta, norma=0.3)
                    predicted_values.append(y_predicted)
                error_on_test = np.sum(loss(fast_get_all_predictions(net, x_test), y))

                predicted_values.append(error_on_test)
                if error_on_test == 0:
                    break

            if predicted_values[-1] == 0:
                mytable1.add_row([number_of_variables, indexes])
                # print("Combination of size ", number_of_variables, " found. Indexes : ", indexes)
                break

    mytable1.float_format = '.3'
    print(f'{mytable1}\n')


def check_combination(indexes: list[int], is_function_binary: bool):
    """
    :param indexes: проверка какого набора производится
    :param is_function_binary: тип ФА
    :return: None
    """
    mytable1 = PrettyTable().copy()
    mytable1.field_names = ["Epoch number", "Вектор весов w", "Выходной вектор y", "Суммарная ошибка Е"]
    x_test, y, centers = initial_function()
    net = Net(is_function_binary, 3, centers)
    errors_in_epoch = []
    norma: float = 0.3
    # обучение НС
    for epoch in range(100):
        for i in indexes:
            x = x_test[i]
            y_true = y[i]
            y_predicted = net.working_mode(x)
            delta = loss(y_true, y_predicted)
            net.studying_mode(delta, norma)
        predicted_values = fast_get_all_predictions(net, x_test)
        error_on_test = np.sum(np.abs(loss(predicted_values, y)))
        errors_in_epoch.append(error_on_test)
        mytable1.add_row([epoch, net.W + [net.bias], predicted_values, error_on_test])
        if error_on_test == 0:
            break

    mytable1.float_format = '.3'
    print(f'{mytable1}\n')
    make_plot(errors_in_epoch, is_function_binary, is_least_combo=True)


if __name__ == '__main__':
    bypass(is_function_binary=True, norma=0.3)  # 1 пункт
    bypass(is_function_binary=False, norma=0.3)  # 2  пункт
    find_least_combination(is_function_binary=True)  # 3  пункт
    check_combination(indexes=[1, 15], is_function_binary=True)
    find_least_combination(is_function_binary=False)  # 4  пункт
    check_combination(indexes=[1, 15], is_function_binary=False)
