import numpy
import matplotlib.pyplot as plt
from prettytable import PrettyTable


def make_plot(values, name: str) -> None:
    """
    рисуем график
    """
    file_name: str = name
    plt.plot(values)
    plt.grid(True)
    plt.title('Error on epoch')
    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.savefig(f'{file_name}.png')
    plt.show()


def Delta(y_true, y_predicted):
    return y_true - y_predicted


def Mean_square_error(y_true, y_predicted):
    return numpy.square(sum(pow(y_true - y_predicted, 2)))


class Level:
    def __init__(self, in_size, out_size, seed=0):
        numpy.random.seed(seed)
        self.W = numpy.random.normal(scale=0.1, size=(out_size, in_size))
        self.b = numpy.random.normal(scale=0.1, size=(out_size))

    def working_mode(self, x):
        self.x = x
        self.net = numpy.dot(self.W, x.transpose()) + self.b
        self.out = (1 - numpy.exp(-self.net)) / (1 + numpy.exp(-self.net))
        return self.out

    def studing_mode(self, delta, norma=1):
        delta = delta * 0.5 * (1 - pow(self.out, 2))
        self.dW = numpy.outer(delta, self.x)
        self.db = delta

        self.next_delta = numpy.dot(delta, self.W)

        self.W = self.W + norma * self.dW
        self.b = self.b + norma * self.db
        return self.next_delta


class NeuralNetwork:

    def __init__(self):
        self.level_1 = Level(1, 1)
        self.level_2 = Level(1, 3)

    def working_mode(self, x):
        self.net = self.level_1.working_mode(x)
        self.net = self.level_2.working_mode(self.net)
        return self.net

    def studing_mode(self, delta, norma):
        delta = self.level_2.studing_mode(delta, norma)
        self.level_1.studing_mode(delta, norma)


if __name__ == '__main__':
    x = numpy.array([-3])
    y = numpy.array([-3 / 10, 1 / 10, 1 / 10])
    x_train = x
    x_test = x

    Network = NeuralNetwork()

    norma: int = 1
    epsilon: float = 1e-5
    errors_on_studing, errors_on_testing, MSE_errors = [], [], []
    MSE: float = epsilon * 100
    epoch_number: int = 1
    table = PrettyTable()
    table.field_names = ['epoch', 'predicted value', 'MSE', 'error']

    while MSE > epsilon:
        y_predicted = Network.working_mode(x_train)
        delta = Delta(y, y_predicted)
        Network.studing_mode(delta, norma)
        errors_on_studing.append(delta)

        y_predicted = Network.working_mode(x_test)
        delta = Delta(y, y_predicted)
        errors_on_testing.append(delta)

        MSE = Mean_square_error(y, y_predicted)
        MSE_errors.append(MSE)

        table.add_row([epoch_number, y_predicted, MSE, delta])
        epoch_number += 1

    table.float_format = '.5'
    print(table)
    make_plot(errors_on_studing, 'errors_on_studing')
    make_plot(errors_on_testing, 'errors_on_testing')
    make_plot(MSE_errors, 'MSE_errors')
