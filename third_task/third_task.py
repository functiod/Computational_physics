import matplotlib.pyplot as plt
import numpy as np
import numdifftools as nd
from scipy import integrate

def f_3a(x: float) -> float:
    return 1 / (1 + pow(x, 2))

def f_3a_2_d() -> nd.Derivative:
    return nd.Derivative(f_3a, n=2)

def f_3a_4_d() -> nd.Derivative:
    return nd.Derivative(f_3a, n=4)

def f_3b(x: float) -> nd.Derivative:
    return x ** (1./3.) * np.exp(np.sin(x))

def f_3b_2_d() -> nd.Derivative:
    return nd.Derivative(f_3a, n=2)

def f_3b_4_d() -> nd.Derivative:
    return nd.Derivative(f_3b, n=4)

def trapese(f: callable, f_d: callable, x_start: float, x_end: float) -> tuple:
    devide_number: int = 10
    x_cur: float = x_start
    x_next: float = x_start
    devide_list: np.ndarray = np.array([2**x for x in range(1, devide_number)])
    sum_list: np.ndarray = np.zeros((devide_number - 1))
    err_list: np.ndarray = np.zeros((devide_number - 1))
    number_points: np.ndarray = np.zeros((devide_number - 1))
    for i, N in enumerate(devide_list):
        step: float = (x_end - x_start) / N
        while x_next != x_end:
            x_next = x_cur + step
            sum_list[i] += 1/2 * (f(x_cur) + f(x_next)) * step
            err_list[i] += -(step**3)/12*f_d(x_cur)
            number_points[i] += 1
            x_cur = x_next
        x_cur: float = x_start
        x_next: float = x_start
    return sum_list, err_list, number_points

def trapese_3b(f: callable, x_start: float, x_end: float) -> tuple:
    devide_number: int = 10
    x_cur: float = x_start
    x_next: float = x_start
    devide_list: np.ndarray = np.array([2**x for x in range(1, devide_number)])
    sum_list: np.ndarray = np.zeros((devide_number - 1))
    number_points: np.ndarray = np.zeros((devide_number - 1))
    for i, N in enumerate(devide_list):
        step: float = (x_end - x_start) / N
        while x_next != x_end:
            x_next = x_cur + step
            sum_list[i] += 1/2 * (f(x_cur) + f(x_next)) * step
            number_points[i] += 1
            x_cur = x_next
        x_cur: float = x_start
        x_next: float = x_start
    return sum_list, number_points

def simpson(f: callable, f_d: callable, x_start: float, x_end: float) -> tuple:
    devide_number: int = 10
    x_cur: float = x_start
    x_next: float = x_start
    devide_list: np.ndarray = np.array([2**x for x in range(1, devide_number)])
    sum_list: np.ndarray = np.zeros((devide_number - 1))
    err_list: np.ndarray = np.zeros((devide_number - 1))
    number_points: np.ndarray = np.zeros((devide_number - 1))
    iter: int = 0
    for i, N in enumerate(devide_list):
        step: float = (x_end - x_start) / N
        while x_next != x_end:
            x_next = x_cur + step
            if iter == 0 or iter == N:
                sum_list[i] += f(x_cur) * step/3.
            elif iter % 2 == 0:
                sum_list[i] += f(x_cur) * step*2/3.
            else:
                sum_list[i] += f(x_cur) * step*4/3.
            err_list[i] += -(step**5)/90.*f_d(x_cur)
            number_points[i] += 1
            x_cur = x_next
            iter += 1
        x_cur: float = x_start
        x_next: float = x_start
    return sum_list, err_list, number_points

def simpson_3b(f: callable, x_start: float, x_end: float) -> tuple:
    devide_number: int = 15
    x_cur: float = x_start
    x_next: float = x_start
    devide_list: np.ndarray = np.array([2**x for x in range(1, devide_number)])
    sum_list: np.ndarray = np.zeros((devide_number - 1))
    number_points: np.ndarray = np.zeros((devide_number - 1))
    iter: int = 0
    for i, N in enumerate(devide_list):
        step: float = (x_end - x_start) / N
        while x_next != x_end:
            x_next = x_cur + step
            if iter == 0 or iter == N:
                sum_list[i] += f(x_cur) * step/3.
            elif iter % 2 == 0:
                sum_list[i] += f(x_cur) * step*2/3.
            else:
                sum_list[i] += f(x_cur) * step*4/3.
            number_points[i] += 1
            x_cur = x_next
            iter += 1
        x_cur: float = x_start
        x_next: float = x_start
    return sum_list, number_points

def plot_err(x: list[list], y: list[list]) -> None:
    x[0] = [abs(x[0][i]) for i in range(len(x[0]))]
    x[1] = [abs(x[1][i]) for i in range(len(x[1]))]
    y[0] = [abs(y[0][i]) for i in range(len(y[0]))]
    y[1] = [abs(y[1][i]) for i in range(len(y[1]))]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlim((min(x[1]), max(x[1])))
    plt.ylim((min(y[1]), max(y[1])))
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.grid(True)
    marker_list = ['*', 'o']
    line_1 = ax.scatter(x[0], y[0], marker=marker_list[0])
    line_2 = ax.scatter(x[1], y[1], marker=marker_list[1])
    ax.legend([line_1, line_2], ['Метод трапеций - 1/N**2', 'Метод Симпсона - 1/N**4'])
    plt.show()

if __name__ == "__main__":
    f_3a_2_deriv: nd.Derivative = f_3a_2_d()
    f_3a_4_deriv: nd.Derivative = f_3a_4_d()
    f_3b_2_deriv: nd.Derivative = f_3b_2_d()
    f_3b_4_deriv: nd.Derivative = f_3b_4_d()

    result_trap_3a: tuple = trapese(f_3a, f_3a_2_deriv, -1.0, 1.0)
    result_simp_3a: tuple = simpson(f_3a, f_3a_4_deriv, -1.0, 1.0)
    result_trap_3b: tuple = trapese_3b(f_3b, 0.0, 1.0)
    result_simp_3b: tuple = simpson_3b(f_3b, 0.0, 1.0)

    x_trap_3a = np.linspace(-1.0, 1.0, 1024)
    x_simp_3a = np.linspace(-1.0, 1.0, 1024)

    theor_result_trap_3a: float = integrate.trapezoid(f_3a(x_trap_3a), x_trap_3a)
    theor_result_simp_3a: float = integrate.simpson(f_3a(x_simp_3a), x_simp_3a)

    x_trap_3b = np.linspace(0.0, 1.0, 1024)
    x_simp_3b = np.linspace(0.0, 1.0, 1024)

    theor_result_trap_3b: float = integrate.trapezoid(f_3b(x_trap_3b), x_trap_3b)
    theor_result_simp_3b: float = integrate.simpson(f_3b(x_simp_3b), x_simp_3b)

    error_trap_3b: list[float] = theor_result_trap_3b - result_trap_3b[0]
    error_simp_3b: list[float] = theor_result_simp_3b - result_simp_3b[0]
    x_err: list[float] = result_trap_3a[2]

    print("Трапеция 3а теор.:", result_trap_3a[0], '\n', "Трапеция 3а теор.:", theor_result_trap_3a)
    print("Симпсмон 3а эксп.:", result_simp_3a[0], '\n', "Трапеция 3а теор.:", theor_result_simp_3a)
    print("Трапеция 3b эксп.:", result_trap_3b[0], '\n', "Трапеция 3b теор.:", theor_result_trap_3b)
    print("Симпсон 3b эксп.:", result_simp_3b[0], '\n', "Трапеция 3b теор.:", theor_result_simp_3b)
    print("Ошибка 3b трапеция:", error_trap_3b)
    print("Ошибка 3b симпсон:", error_simp_3b)

    plot_err([x_err, x_err], [result_trap_3a[1], result_simp_3a[1]])
