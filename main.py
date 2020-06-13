import numpy as np
import time


class Minimize:
    def __init__(self, A, x_hat, eps=1e-5):
        N, M = np.shape(A)
        self.A = A
        self.x_hat = x_hat
        self.b = np.dot(A, x_hat) + np.random.randn(N, 1) * np.linalg.norm(A) * 1e-5
        self.fx_hat = self.func(self.x_hat)
        self.eps = eps
        self.AA = np.dot(A.T, A)
        hesse = 2 * self.AA
        lmd, _ = np.linalg.eig(hesse)
        L = np.max(lmd)
        self.alpha = 1 / L

    def func(self, x):
        norm = np.linalg.norm(self.b - np.dot(self.A, x))
        return norm * norm

    def diffunc(self, x):
        return 2 * np.dot(self.AA, x) - 2 * np.dot((self.A).T, self.b)

    def gradient_discent(self, x):
        fx = self.func(x)
        while abs(fx - self.fx_hat) > self.eps:
            x = x - self.alpha * self.diffunc(x)
            fx = self.func(x)

    def polyak_momentum(self, x):
        fx = self.func(x)
        prev_x = x
        k = 0
        while abs(fx - self.fx_hat) > self.eps:
            beta = k / (k + 3)
            k += 1
            tmp = x
            x = x - self.alpha * self.diffunc(x) + beta * (x - prev_x)
            prev_x = tmp
            fx = self.func(x)

    def nesterov_acceleration(self, x):
        fx = self.func(x)
        y = x
        k = 0
        while abs(fx - self.fx_hat) > self.eps:
            beta = k / (k + 3)
            k += 1
            prev_x = x
            x = y - self.alpha * self.diffunc(y)
            y = x + beta * (x - prev_x)
            fx = self.func(x)


if __name__ == "__main__":
    for M in [10, 50, 100, 200, 500, 800, 1000]:
        N = int(M * 1.5)
        A = np.random.rand(N, M)
        x_hat = np.ones((M, 1))
        minimize = Minimize(A, x_hat)
        x = 1 + np.random.rand(M, 1) * 0.1

        start = time.time()
        minimize.gradient_discent(x)
        t = time.time() - start
        print('[(N, M) = ({}, {})] gradient_discent {} s'.format(N, M, t))

        start = time.time()
        minimize.polyak_momentum(x)
        t = time.time() - start
        print('[(N, M) = ({}, {})] polyak_momentum {} s'.format(N, M, t))

        start = time.time()
        minimize.nesterov_acceleration(x)
        t = time.time() - start
        print('[(N, M) = ({}, {})] nesterov_acceleration {} s'.format(N, M, t))
