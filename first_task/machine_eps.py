import numpy as np
import math
import sys

def find_eps_32() -> np.float32:
    eps: np.float32 = np.float32(1.0)
    while np.float32(1.0) + eps / np.float32(2.0) > np.float32(1.0):
        eps /= np.float32(2.0)
    return eps

def find_eps_64() -> float:
    eps: float = 1.0
    while 1.0 + eps / 2.0 > 1.0:
        eps /= 2.0
    return eps

def find_number_digit(eps: float) -> int:
    result: float = eps
    my_buff: list = []
    while result <= 1.0:
        result *= 2
        my_buff.append(1)
    return len(my_buff)

def min_max_value(degree: int) -> tuple:
    min_result: float = pow(-2, degree - 1) + 2
    max_result: float = pow(2, degree - 1) - 1
    return min_result, max_result

def compare(eps: float) -> list:
    list_to_compare: list = [1.0, 1.0 + eps, 1.0 + eps / 2, 1.0 + eps + eps / 2]
    result_list: list = []
    for i in range(len(list_to_compare)):
        for j in range(i + 1, len(list_to_compare)):
            if list_to_compare[i] > list_to_compare[j]:
                result_list.append(str(list_to_compare[i]) + ">" + str(list_to_compare[j]))
            elif list_to_compare[i] == list_to_compare[j]:
                result_list.append(str(list_to_compare[i]) + "=" + str(list_to_compare[j]))
            else:
                result_list.append(str(list_to_compare[i]) + "<" + str(list_to_compare[j]))
    return result_list

def find_max_degree(dig: int) -> int:
    degree: int = 64 - dig
    return pow(2, degree - 1) - 1

def find_min_degree(dig: int) -> int:
    degree: int = 64 - dig
    return -pow(2, degree - 1) + 2

if __name__ == "__main__":
    # print(sys.float_info)
    # print(find_eps_64())
    # print(find_number_digit(find_eps_64()))
    # print(find_max_degree(find_number_digit(find_eps_64())))
    # print("digits_number_64: ", find_number_digit(find_eps_64()))
    # print("comparing_64:", compare(find_eps_64()))
    # print(1.0 + eps/2 + eps)
    # print(eps, eps/2, eps/4)
    # print(min_max_value(find_number_digit(find_eps_64())))
    # Начальное приближение
