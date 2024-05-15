import numpy as np
from numpy import dot, outer, sqrt
from numpy.linalg import eig
import scipy


class CMA_ES:
    def __init__(self,
                 population=10, num_parents=5, sample_dim=100, step_length=0.001):
        self.N = population
        self.M = num_parents
        self.sigma = step_length
        self.C = np.eye(sample_dim)
        self.dim = sample_dim

    def generate(self):
        x = np.random.randn(self.N, self.dim) * self.sigma * 0.
        return x

    def step(self, x, loss):
        """
        使用样本x和损失loss更新协方差矩阵。
        :param x: array [population, dim]
        :param loss: array [population]
        :return:
        """
        weights = np.log(self.M + 1 / 2) - np.log(np.arange(1, self.M + 1))  # [M]
        weights = weights / np.sum(weights)
        mu_eff = np.sum(weights) ** 2 / np.sum(weights ** 2)

        ar_index = np.argsort(loss)  # [N]
        ar_x = x[ar_index[:self.M]]  # [M, dim]
        x_old = np.sum(weights*ar_x.T, axis=1)  # [dim]
        z_mean = ar_x - x_old

        artmp = (1 / self.sigma) * z_mean
        c1 = 2 / ((self.dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2) ** 2 + mu_eff))
        C = (1 - c1 - cmu) * self.C + c1 * np.outer(x_old, x_old) + cmu * dot(artmp.T, dot(np.diag(weights), artmp))
        self.C = C

        # 生成新一代种群
        D, B = eig(C)
        D = sqrt(D)
        for i in range(self.N):
            x[i, :] = x_old + self.sigma * dot(B, D * np.random.randn(self.dim))

        return x

