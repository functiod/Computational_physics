import matplotlib.pyplot as plt
import numpy as np

eps: float = 0.0001

def f(x: float) -> float:
    return 1/np.tan(1 + x) - np.sqrt(1/x + 1)

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

def dihotomy(start: float, end: float) -> float:
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
    return end_1

def iter_f(x: float) -> float:
    return pow(np.tan(1+x), 2) / (1 - pow(np.tan(1+x), 2))

def iteration(x: float) -> float:
    x_next: float
    x_cur: float = x
    while True:
        x_next = iter_f(x_cur)
        if abs(x_next - x_cur) < eps:
            return x_next
        x_cur = x_next

def plot_func() -> None:
    t: np.NDArray = np.arange(-10.0, 10.0, 0.01)
    y = pow(np.tan(1+t), 2) / (1 - pow(np.tan(1+t), 2))
    # y = 1/np.tan(1 + t) - np.sqrt(1/t + 1)
    plt.xticks(np.arange(-10.0, 10.0, 1))
    plt.ylim((-10, 10))
    plt.grid(True)
    plt.plot(t, y, "-")
    plt.show()

def f(x: float) -> float:
    return 1/np.tan(1 + x) - np.sqrt(1/x + 1)

def deriv_f(x: float) -> float:
    return 1/2*pow(x, 2) * 1/np.sqrt(1+ 1/x) - 1/pow(np.sin(1+x), 2)

def newton_method(x_start: float, eps: float) -> tuple:
    result: float = 1.0
    x_cur: float = x_start
    x_next: float
    while abs(result) >= eps:
        x_next = x_cur - f(x_cur) / deriv_f(x_cur)
        x_cur = x_next
        result = f(x_next)
    return x_next, result

print(newton_method(-7.0, 0.0001))


if __name__ == "__main__":
    start: float = -4.0
    end: float = -2.0
    # if solution_exist(start, end):
    #     print("Xo =", dihotomy(start, end))
    #     print("f(Xo) =", f(dihotomy(start, end)))
    # else:
    #     print('choose another values')
    print(iteration(-3.0))
    plot_func()
