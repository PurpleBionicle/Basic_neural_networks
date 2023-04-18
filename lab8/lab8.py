import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
import random


def show_signal(signal):
    signal_binary = np.array([[0 if element == -1 else 1 for element in row] for row in signal])
    plt.imsave('lab8/1.png', np.array(signal_binary[0]).reshape(5, 5), cmap='Greens')
    plt.imsave('lab8/2.png', np.array(signal_binary[1]).reshape(5, 5), cmap='Greens')
    plt.imsave('lab8/3.png', np.array(signal_binary[2]).reshape(5, 5), cmap='Greens')
    pics = [mpimg.imread('lab8/1.png'), mpimg.imread('lab8/2.png'), mpimg.imread('lab8/3.png')]
    positions = [331, 332, 333]
    plt.subplot(positions[0])
    plt.imshow(pics[0])
    plt.subplot(positions[1])
    plt.imshow(pics[1])
    plt.subplot(positions[2])
    plt.imshow(pics[2])
    plt.show()


def initial_function():
    # пары ассоциации (отклик)
    answer = np.array([[1, 1, 1], [1, -1, 1], [1, -1, -1]])
    # 5x5
    signal = np.array([
        [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1],
        [1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1]]
    )
    return answer, signal


def working_mode(weights):
    def activation_function(net, last_result):
        for neurons_number in range(len(net)):
            for index_in_neuron in range(len(net[neurons_number])):
                if net[neurons_number, index_in_neuron] == 0:
                    net[neurons_number, index_in_neuron] = last_result[neurons_number, index_in_neuron]
                else:
                    net[neurons_number, index_in_neuron] = -1 if net[neurons_number, index_in_neuron] < 0 else 1
        return net

    def get_net_forward_bypass(signal, weight):
        return np.dot(signal, weight)

    def get_net_backward_bypass(response, weight):
        return np.dot(response, weight.transpose())

    def out_forward_bypass(signal, weight, expected_signal):
        # получение ответа
        net = get_net_forward_bypass(signal, weight)
        response_got = activation_function(net, expected_signal)
        return response_got

    def out_backward_bypass(response, weight, expected_response):
        # получение сигнала
        net = get_net_backward_bypass(response, weight)
        signal_got = activation_function(net, expected_response)
        return signal_got

    # активируем тремя образами ее первый слой
    response_got = out_forward_bypass(signal, weights, signal)  # расчет входов и выходов второго слоя
    signal_got = out_backward_bypass(response_got, weights, answer)  # расчет входов и выходов первого слоя
    show_signal(signal_got)


def make_some_noise(noisy_signal, noise_koeff):
    pixels_count = noisy_signal.shape[1]
    #  найдем засоренные индексы
    indexes = [random.randint(0, pixels_count * noisy_signal.shape[0])
               for _ in range(int(pixels_count * noise_koeff) * noisy_signal.shape[0])]
    for index in indexes:
        div, mod = index // pixels_count, index % pixels_count
        noisy_signal[div, mod] *= (-1)

    show_signal(noisy_signal)
    return noisy_signal


if __name__ == '__main__':
    answer, signal = initial_function()


    show_signal(signal)

    # Настроим веса РНС Коско
    weights = np.dot(signal.transpose(), answer)
    working_mode(weights)

    # Добавим шум
    signal_with_noise = make_some_noise(signal.copy(), 0.1)
    # прогоним новый сигнал через сетку
    working_mode(weights)
