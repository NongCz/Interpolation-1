import numpy as np
import matplotlib.pyplot as plt
import random
import os

a, b, m = 0, 4, 5
CUR_DIR = os.path.join(os.getcwd(), "Day8", "interpolation&lagrange", "images")
os.makedirs(CUR_DIR, exist_ok=True)  

def f(x):
    return np.sin(5 * x)

def get_matrix(points_x):
    return [[points_x[i]**j for j in range(m)] for i in range(m)]

def interpolate(x, solve_matrix):
    return sum(solve_matrix[j] * (x ** j) for j in range(m))

def get_L(exc, points_x, x):
    res = 1
    for i in range(m):
        if i != exc:
            res *= (x - points_x[i]) / (points_x[exc] - points_x[i])
    return res

def lagrange(x, points_x, real_answer):
    return sum(get_L(i, points_x, x) * real_answer[i] for i in range(m))

points_x = [random.uniform(a, b) for _ in range(m)]
real_answer = [f(x) for x in points_x]

matrix = get_matrix(points_x)
solve_matrix = np.linalg.solve(matrix, real_answer)

interp_x = np.linspace(a, b, 100)
interp_y = np.array([interpolate(x, solve_matrix) for x in interp_x])
lagrange_y = np.array([lagrange(x, points_x, real_answer) for x in interp_x])
real_graph = f(interp_x)

plt.figure(figsize=(10, 6))
plt.plot(points_x, real_answer, 'ro', label='Real Points')
plt.plot(interp_x, real_graph, label='Real Function')
plt.plot(interp_x, lagrange_y, label='Lagrange Function')
plt.plot(interp_x, interp_y, '--', label='Interpolation')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Comparison of Real Function and Interpolation f(x) = sin(5x), m={m} points')
plt.legend()
plt.grid(True)

img_name = "plot_sinX.png"
plt.savefig(os.path.join(CUR_DIR, img_name))
plt.show()