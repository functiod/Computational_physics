import matplotlib.pyplot as plt
import numpy as np
from time import sleep

eps: float = 0.0001

def f(x: float) -> float:
    return 1/np.tan(1 + x) - np.sqrt(1/x + 1)
    # return np.cos(x) - pow(x, 2)

def deriv_f(x: float) -> float:
    return 1./2. * 1/(pow(x, 2) * np.sqrt(1 + 1/x)) - (1 + 1/pow(np.tan(1 + x), 2))
    # return -np.sin(x) - 2*x

# ______________________________FIRST PART_______________________________
def solution_exist(start: float, end: float) -> bool:
    result: bool
    if f(start) * f(end) > 0:
        result = False
    else:
        result = True
    return result

def choose_step(start: float, end: float) -> bool:
    result: bool
    if f(start) * f((start + end) / 2) <= 0:
        result = True
    else:
        result = False
    return result

def dihotomy(start: float, end: float) -> tuple:
    iter: int = 0
    result: float = 1.0
    start_1: float = start
    end_1: float = end
    while abs(result) >= eps:
        if choose_step(start_1, end_1):
            start_1 = start_1
            end_1 = (start_1 + end_1) / 2
        else:
            start_1 = (start_1 + end_1) / 2
            end_1 = end_1
        result = f(end_1)
        iter += 1
    return end_1, f(end_1), iter

# ______________________________SECOND PART_______________________________

def iter_f(x: float) -> float:
    # return np.sqrt((x + x*x)) * np.tan(1+x)
    return x + f(x)

def deriv__iter_f(x: float) -> float:
    # return np.sqrt((x + x*x)) / (1 + pow((1 + x), 2)) + np.tan(1 +x )*(1 + 2*x) / (2 * np.sqrt((x + x*x)))
    return 1 - deriv_f(x)

def converge_f(x: float) -> float:
    result_f: float
    if abs(deriv__iter_f(x)) > 1:
        result_f = 1/iter_f(x)
    else:
        result_f = iter_f(x)
    return result_f

def iteration_1(x: float) -> tuple:
    x_next: float
    x_cur: float = x
    iter: int = 0
    while True:
        iter += 1
        x_next = iter_f(x_cur)
        if abs(x_next - x_cur) <= eps or abs(f(x_next)) <= eps:
            return x_cur, x_next, f(x_next), iter
        x_cur = x_next

def iteration_2(x: float) -> tuple:
    x_next: float
    x_cur: float = x
    iter: int = 0
    while iter < 1000:
        iter += 1
        x_next = converge_f(x_cur)
        x_cur = x_next
        if abs(f(x_next)) <= eps:
            return x_cur, x_next, f(x_next), iter, 'hello'
    return x_next, f(x_next), iter

# ______________________________THIRD PART_______________________________

def newtone_method(x_start: float) -> tuple:
    x_cur: float = x_start
    x_next: float
    iter: int = 0
    deriv_list: list = []
    while abs(f(x_cur)) >= eps:
        deriv_list.append(deriv_f(x_cur))
        iter += 1
        x_next = x_cur - f(x_cur) / deriv_f(x_cur)
        x_cur = x_next
    return x_cur, f(x_cur), iter

# ______________________________VISUAL PART_______________________________

def plot_func() -> None:
    t: np.NDArray = np.arange(-10.0, 10.0, 0.01)
    # y = 1./2. * 1/(pow(t, 2) * np.sqrt(1 + 1/t)) - (1 + 1/pow(np.tan(1 + t), 2))
    # y = 1/np.tan(1 + t) - np.sqrt(1/t + 1)
    y = np.cos(t) - pow(t, 2)
    plt.xticks(np.arange(-10.0, 10.0, 1))
    plt.ylim((-10, 10))
    plt.grid(True)
    plt.plot(t, y, "-")
    plt.show()

def visualize_newtone(x_start: float) -> None:
    t = np.arange(-10.0, 10.0, 0.01)
    y = 1/np.tan(1 + t) - np.sqrt(1/t + 1)
    y_deriv = 1./2. * 1/(pow(t, 2) * np.sqrt(1 + 1/t)) - (1 + 1/pow(np.tan(t), 2))
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xticks(np.arange(-10.0, 10.0, 1))
    plt.xlim((-10, 10))
    plt.ylim((-10, 10))
    plt.grid(True)
    x_cur: float = x_start
    x_next: float
    line1 = ax.plot(t, y)
    line2 = ax.plot(t, y_deriv)
    dot1 = ax.scatter(x_cur, deriv_f(x_cur), c = 'red', marker = '*')
    dot2 = ax.scatter(x_cur, f(x_cur), c = 'green', marker = 'o')
    while abs(f(x_cur)) >= eps:
        x_next = x_cur - f(x_cur) / deriv_f(x_cur)
        x_cur = x_next
        dot1.set_offsets((x_cur ,deriv_f(x_cur)))
        dot2.set_offsets((x_cur, f(x_cur)))
        fig.canvas.draw()
        fig.canvas.flush_events()
        sleep(1)


if __name__ == "__main__":
    start: float = -7.0
    end: float = -5.0
    # plot_func()
    if solution_exist(start, end):
        print('dihotomy', dihotomy(start, end))
    else:
        print('choose another values')
    print("newtone_method:", newtone_method(-7.0))
    print("simple iteration:", iteration_1(-3.0))
