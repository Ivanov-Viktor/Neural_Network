import numpy as np


def f(x):
    return 2 / (1 + np.exp(-x)) - 1


def df(x):
    return 0.5 * (1 + x) * (1 - x)


w1 = np.array([[-0.2, 0.3, -0.4], [0.1, -0.3, -0.4]])
w2 = np.array([0.2, 0.3])


def go_forward(inp):
    summa = np.dot(w1, inp)
    out = np.array([f(x) for x in summa])

    summa = np.dot(w2, out)
    y = f(summa)
    return y, out


def train(epoch):
    global w2, w1
    lmd = 0.01  # шаг обучения
    n = 10000  # число итераций при обучении
    count = len(epoch)
    for k in range(n):
        x = epoch[np.random.randint(0, count)]  # случайный выбор входного сигнала из обучающей выборки
        y, out = go_forward(x[0:3])  # прямой проход по НС и вычисление входныч значений нерона
        e = y - x[-1]  # ошибка
        delta = e * df(y)  # локальный градиент
        w2[0] = w2[0] - lmd * delta * out[0]  # корректировка веса первой связи
        w2[1] = w2[1] - lmd * delta * out[1]  # корректировка веса второй связи

        delta2 = w2 * delta * df(out)  # вектор из 2-х величин локальных градиентов

        # корректировка связей первого слоя
        w1[0, :] = w1[0, :] - np.array(x[0:3]) * delta2[0] * lmd
        w1[1, :] = w1[1, :] - np.array(x[0:3]) * delta2[1] * lmd


# обучающая выборка (она же полная выборка)
epoch = [(-1, -1, -1, -1),
         (-1, -1, 1, 1),
         (-1, 1, -1, -1),
         (-1, 1, 1, 1),
         (1, -1, -1, -1),
         (1, 1, -1, -1),
         (1, 1, -1, -1),
         (1, 1, 1, -1)]

train(epoch)  # запуск обучения сети

for x in epoch:
    y, out = go_forward(x[0:3])
    print(f"выходное значение НС: {y} => {x[-1]}")
