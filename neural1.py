import numpy as np


def act(x):
    return 0 if x <= 0 else 1


def go(c):
    x = np.array([c[0], c[1], 1])
    w1 = [1, 1, -1.5]
    w2 = [1, 1, -0.5]
    w_hidden = np.array([w1, w2])
    w_out = np.array([-1, 1, -0.5])

    summa = np.dot(w_hidden, x)
    out = [act(x) for x in summa]
    out.append(1)
    out = np.array(out)

    summa = np.dot(w_out, out)
    y = act(summa)
    return y


C1 = [(1, 0), (0, 1)]
C2 = [(0, 0), (1, 1)] 

print(go(C1[0]), go(C1[1]))
print(go(C2[0]), go(C2[1]))
