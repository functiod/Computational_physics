import matplotlib.pyplot as plt
import numpy as np

def f_3a(x: float) -> float:
    return 1 / (1 + pow(x, 2))

def f_3a_d_deriv(x: float) -> float:
    return -2. / (1 + x**2)**2 + 8*x**2 / (1 + x**2)**3

def f_3b(x: float) -> float:
    return x ** (1./3.) * np.exp(np.sin(x))

def trapese(f: float, x_start: float = -1.0, x_end: float = 1.0) -> tuple:
    x_cur: float = x_start
    x_next: float = x_start
    devide_list: np.ndarray = np.array([4, 8, 16, 32, 64])
    sum_list: np.ndarray = np.zeros((5))
    err_list: np.ndarray = np.zeros((5))
    for i, N in enumerate(devide_list):
        step: float = (x_end - x_start) / N
        while x_next != x_end:
            x_next = x_cur + step
            sum_list[i] += 1/2 * (f(x_cur) + f(x_next)) * step
            err_list[i] += -(step**3)/12.*f_3a_d_deriv(x_cur)
            x_cur = x_next
        x_cur: float = x_start
        x_next: float = x_start
    return sum_list, err_list

if __name__ == "__main__":
    print(trapese(f_3a, -1.0, 1.0))