import math
from itertools import combinations
from typing import List, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
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
    plt.savefig(f'lab1/{file_name}.png')
    plt.show()


def boolean_function(x: list[int]) -> int:
    """
    :param x: набор значений переменных
    :return: результат функции
    """
    return int((x[0] or x[1]) and x[2] and x[3])


def initial_function() -> tuple[list[list[int] | Any], list[int]]:
    """
    :return: наборы значений переменных и значение функции
    """
    x = [[0, 0, 0, 0, 1], [0, 0, 0, 1, 1], [0, 0, 1, 0, 1], [0, 0, 1, 1, 1],
         [0, 1, 0, 0, 1], [0, 1, 0, 1, 1], [0, 1, 1, 0, 1], [0, 1, 1, 1, 1],
         [1, 0, 0, 0, 1], [1, 0, 0, 1, 1], [1, 0, 1, 0, 1], [1, 0, 1, 1, 1],
         [1, 1, 0, 0, 1], [1, 1, 0, 1, 1], [1, 1, 1, 0, 1], [1, 1, 1, 1, 1]]
    return x, [boolean_function(x[i]) for i in range(len(x))]


def working_mode(is_function_binary: bool, x: list[int], weights: list[float]):
    """
    :param is_function_binary: тип
    :param x: значения иксов
    :param weights: текущие веса
    :return: предсказанные значения
    """

    def get_net(x: list[int], weights: list[float]):
        net = 0
        for i in range(len(x)):
            net += x[i] * weights[i]
        return net

    net = get_net(x, weights)
    out: float = 0
    if is_function_binary:
        t: int = 1 if net >= 0 else 0

    else:
        out = 1 / (1 + math.exp(-net))
        t: int = 1 if out >= 0.5 else 0
    return net, out, t


def count_error(t, y):
    return t - y


def studying_mode(weights: list, net: float, delta: int, x_set: list, k: float, threshold_func: bool):
    """
        Корректировка весов синопсов
    """
    if threshold_func:
        for i in range(len(weights)):
            weights[i] -= k * delta * x_set[i]
        return weights
    else:
        for i in range(len(weights)):
            weights[i] -= k * delta * (1 / (1 + math.exp(-net)) * (1 - 1 / (1 + math.exp(-net)))) * x_set[i]
        return weights


"""ПУНКТ 1.2"""


def bypass(is_function_binary: bool):
    """
    Осуществляет проход нейросети с выбранной функцией активации
    :param is_function_binary: тип функции активации
    :return: NONE
    """
    mytable1 = PrettyTable().copy()
    mytable1.field_names = ["Epoch number", "Вектор весов w", "Выходной вектор y", "Суммарная ошибка Е"]
    x, y = initial_function()
    print(f'таблица истинности булевой функции : {y}')
    norma: float = 0.3
    weights: list[float] = np.zeros(5)
    errors_in_epoch: list[int] = []
    error_in_current_epoch = len(y)
    epoch: int = 0

    while error_in_current_epoch > 0:
        error_in_current_epoch = 0
        predicted = []
        for i in range(len(x)):
            net, out, t = working_mode(is_function_binary, x[i], weights)
            predicted.append(t)
            delta = t - y[i]
            if delta != 0:
                error_in_current_epoch += 1
            weights = studying_mode(weights, net, delta, x[i], norma, is_function_binary)

        errors_in_epoch.append(error_in_current_epoch)
        mytable1.add_row([epoch, weights, predicted, error_in_current_epoch])
        epoch += 1

    mytable1.float_format = '.3'
    print(f'{mytable1}\n')
    make_plot(errors_in_epoch, is_function_binary)


def check_error(weights: list[float], is_function_binary: bool):
    x, y = initial_function()
    error: int = 0
    predicted: list[int] = []
    for i in range(len(y)):
        net, out, t = working_mode(is_function_binary, x[i], weights)
        predicted.append(t)
        delta = t - y[i]
        if delta != 0:
            error += 1
    return error


"""ПУНКТ 3.4"""


def find_least_combination(is_function_binary: bool):
    """
    нахождение минимального обущающего набора векторов
    """
    print('Combinations for binary FA' if is_function_binary else 'Combinations for softsign FA')
    mytable1 = PrettyTable().copy()
    mytable1.field_names = ["length of vectors", "combination of vectors"]
    x, y = initial_function()
    norma: float = 0.3
    for variables in range(len(y) - 1, 0, -1):
        combs = list(combinations(range(len(y)), variables))
        for indexes in combs:
            weights: list[float] = np.zeros(5)
            while True:
                error_in_current_epoch: int = 0
                predicted: list[int] = []
                for i in indexes:
                    net, out, t = working_mode(is_function_binary, x[i], weights)
                    predicted.append(t)
                    delta = t - y[i]
                    if delta != 0:
                        error_in_current_epoch += 1
                    weights = studying_mode(weights, net, delta, x[i], norma, is_function_binary)

                final_error = error_in_current_epoch
                if final_error == 0:
                    break

            error_set = check_error(weights=weights, is_function_binary=is_function_binary)

            if error_set == 0:
                mytable1.add_row([variables, indexes])
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
    x, y = initial_function()
    norma: float = 0.3
    errors_in_epoch: list[int] = []
    error: int = len(y)
    epoch: int = 0
    weights: list[float] = np.zeros(5)
    # обучение НС
    while error > 0:
        error = 0
        predicted = []
        for i in indexes:
            net, out, t = working_mode(is_function_binary, x[i], weights)
            delta = count_error(t, y[i])
            weights = studying_mode(weights, net, delta, x[i], norma, is_function_binary)

        for i in range(16):
            net, out, t = working_mode(is_function_binary, x[i], weights)
            predicted.append(t)
            delta = count_error(t, y[i])
            if delta != 0:
                error += 1

        final_error = error
        errors_in_epoch.append(final_error)
        mytable1.add_row([epoch, weights, predicted, error])
        epoch += 1

    mytable1.float_format = '.3'
    print(f'{mytable1}\n')
    make_plot(errors_in_epoch, is_function_binary, is_least_combo=True)


def main():
    # 1,3 - функции активации
    bypass(is_function_binary=True)
    bypass(is_function_binary=False)
    find_least_combination(is_function_binary=True)
    check_combination(indexes=[2, 13, 15], is_function_binary=True)
    find_least_combination(is_function_binary=False)
    check_combination(indexes=[1, 7, 10, 11], is_function_binary=False)


if __name__ == '__main__':
    main()
