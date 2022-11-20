import numpy as np


def act(x):
    return 0 if x < 0.5 else 1


def go([house, rock, attr]):
    x = np.array([house, rock, attr])
    w11 = [0.3, 0.3, 0]
    w12 = [0.4, -0.5, 1]
    weight1 = np.array([w11, w12])  # матрица 2х3
    weight2 = np.array([-1, 1])     # вектор 1х3

    sum_hidden = np.dot(weight1, x) # вычисляем сумму на входах нейронов скрытого слоя
    print("Значение сумм на нейронах скрытого слоя:" + str(sum_hidden))

    out_hidden = np.array([act(x) for x in sum_hidden])
    print("значение ны выходах нейронов скрытого слоя: " + str(out_hidden))

    sum_end