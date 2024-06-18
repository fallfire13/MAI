import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import circulant

import numpy as np



def arnoldi_iteration(A, b, n):
    m = A.shape[0]

    h = np.zeros((n + 1, n), dtype=np.complex)
    Q = np.zeros((m, n + 1), dtype=np.complex)

    q = b / np.linalg.norm(b)  # Нормализация входного вектора
    Q[:, 0] = q  # Использование его как первого вектора Крылова

    for k in range(n):
        v = A.dot(q)  # Генерация нового вектора
        for j in range(k + 1): 
            h[j, k] = np.dot(Q[:, j], v)
            v = v - h[j, k] * Q[:, j]

        h[k + 1, k] = np.linalg.norm(v)
        eps = 1e-12  
        if h[k + 1, k] > eps:  # Добавление полученного вектора в список, если только
            q = v / h[k + 1, k]  # получается нулевой вектор.
            Q[:, k + 1] = q
        else:  # Если это произойдет. Прекратить повторять
            return Q, h
    return Q, h

# Построение матрицы А
N = 2**4
I = np.eye(N)
k = np.fft.fftfreq(N, 1.0 / N) + 0.5
alpha = np.linspace(0.1, 1.0, N)*2e2
c = np.fft.fft(alpha) / N
C = circulant(c)
A = np.einsum("i, ij, j->ij", k, C, k)

# Показываем, что A является эрмитовым
print(np.allclose(A, A.conj().T))

# Произвольный (случайный) начальный вектор
np.random.seed(0)
v = np.random.rand(N)
# Выполнение итерации Арнольди с комплексным А
_, h = arnoldi_iteration(A, v, N)
# Выполнение итерации Арнольди с вещественным A
_, h2 = arnoldi_iteration(np.real(A), v, N)

print(_)
# Построение результатов.
plt.subplot(121)
plt.imshow(np.abs(h))
plt.title("Complex A")
plt.subplot(122)
plt.imshow(np.abs(h2))
plt.title("Real A")
plt.tight_layout()
plt.show()
