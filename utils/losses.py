import numpy as np


def forward_loss(start_point, end_point, direction):
    """
    :param start_point: world pos [3]
    :param end_point:
    :param direction:  world vec [3]
    :return: float
    """
    vec = direction / np.linalg.norm(direction, ord=2)
    loss = - np.dot(vec, (end_point - start_point))
    loss = np.sum(loss)
    return loss


def upright_loss(t, max_t):
    loss = (max_t - t) ** 2
    return loss


def smoothness_loss(acc):
    """
    :param acc: [T, 3]
    :return:
    """
    loss = np.sum(np.dot(acc, acc.T))
    return loss

