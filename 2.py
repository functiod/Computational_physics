import matplotlib.pyplot as plt
import numpy as np

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