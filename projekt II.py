import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Dane
data = np.array([[1.0, 3.0], [2.0, 1.0], [3.5, 4.0], [5.0, 0.0], [6.0, 0.5], [9.0, -2.0], [9.5, -3.0]])

# Zad.1: Naturalna funkcja sklejana sześcienna
def natural_cubic_spline(nodes):
    n = len(nodes) - 1
    h = np.diff(nodes[:, 0])
    b = 6 * np.diff(nodes[:, 1]) / h
    
    u = np.zeros(n)
    v = np.zeros(n)
    
    u[1] = 2 * (h[0] + h[1])
    v[1] = b[1] - b[0]
    
    for i in range(2, n):
        u[i] = 2 * (h[i-1] + h[i]) - (h[i-1]**2) / u[i-1]
        v[i] = b[i] - b[i-1] - h[i-1] * v[i-1] / u[i-1]
    
    z = np.zeros(n + 1)
    z[-1] = 0
    
    for i in range(n - 1, 0, -1):
        z[i] = (v[i] - h[i] * z[i+1]) / u[i] 
    
    A = np.zeros(n)
    B = np.zeros(n)
    C = np.zeros(n)
    
    for i in range(n):
        A[i] = (z[i+1] - z[i]) / (6 * h[i])
        B[i] = z[i] / 2
        C[i] = -(h[i] * z[i+1] + 2 * h[i] * z[i]) / 6 + (nodes[i+1, 1] - nodes[i, 1]) / h[i]
    
    return A, B, C

def cubic_spline_interpolation(x, nodes, A, B, C):
    n = len(nodes) - 1
    S = np.zeros(x.shape[0])
    
    for i in range(n):
        mask = np.logical_and(x >= nodes[i, 0], x <= nodes[i+1, 0])
        S[mask] = nodes[i, 1] + (x[mask] - nodes[i, 0]) * (C[i] + (x[mask] - nodes[i, 0]) * (B[i] + (x[mask] - nodes[i, 0]) * A[i]))
    
    S[x == nodes[-1, 0]] = nodes[-1, 1]
    
    return S

# Obliczanie współczynników funkcji sklejanej
A, B, C = natural_cubic_spline(data)

# Wykres funkcji sklejanej
x_interpolation = np.linspace(data[0, 0], data[-1, 0], 1000)
S_interpolation = cubic_spline_interpolation(x_interpolation, data, A, B, C)

plt.plot(data[:, 0], data[:, 1], 'ro', label='Data Points')
plt.plot(x_interpolation, S_interpolation, label='Custom Cubic Spline')
plt.legend()
plt.xlabel('x')
plt.ylabel('S(x)')
plt.title('Natural Cubic Spline Interpolation')
plt.grid(True)
plt.show()

# Zad.2: Funkcja sklejana stopnia 1
def linear_interpolation(x, nodes):
    N = len(nodes) - 1
    P_linear = np.zeros(x.shape)
    
    for n in range(N):
        if n == 0:
            P_linear += ((nodes[n+1, 1] - nodes[n, 1]) / (nodes[n+1, 0] - nodes[n, 0]) * (x - nodes[n, 0]) + nodes[n, 1]) * (x <= nodes[n+1, 0])
        elif n == N-1:
            P_linear += ((nodes[n+1, 1] - nodes[n, 1]) / (nodes[n+1, 0] - nodes[n, 0]) * (x - nodes[n, 0]) + nodes[n, 1]) * (x > nodes[n, 0])
        else:
            P_linear += ((nodes[n+1, 1] - nodes[n, 1]) / (nodes[n+1, 0] - nodes[n, 0]) * (x - nodes[n, 0]) + nodes[n, 1]) * ((x > nodes[n, 0]) & (x <= nodes[n+1, 0]))
    
    return P_linear

# Zad.2: Wielomian interpolacyjny Lagrange'a
def lagrange_interpolation(x, nodes):
    N = len(nodes)
    lagrange_poly = np.ones((N, x.shape[0]))
    
    for i in range(N):
        for j in range(N):
            if j != i:
                lagrange_poly[i, :] *= (x - nodes[j, 0]) / (nodes[i, 0] - nodes[j, 0])
    
    p = np.zeros(x.shape[0])
    
    for n in range(N):
        p += lagrange_poly[n, :] * nodes[n, 1]
    
    return p

# Zad.2: Porównanie funkcji sklejanej sześciennej, funkcji sklejanej stopnia 1 i wielomianu interpolującego Lagrange'a
x_interpolation_zad2 = np.linspace(data[0, 0], data[-1, 0], 1000)
P_linear = linear_interpolation(x_interpolation_zad2, data)
P_lagrange = lagrange_interpolation(x_interpolation_zad2, data)

plt.plot(data[:, 0], data[:, 1], 'ro', label='Data Points')
plt.plot(x_interpolation_zad2, S_interpolation, label='Natural Cubic Spline')
plt.plot(x_interpolation_zad2, P_linear, label='Linear Interpolation')
plt.plot(x_interpolation_zad2, P_lagrange, label='Lagrange Interpolation')
plt.legend()
plt.xlabel('x')
plt.ylabel('Interpolated Values')
plt.title('Comparison of Interpolation Methods')
plt.grid(True)
plt.show()

# Zad.3: Porównanie funkcji sklejanej sześciennej i CubicSpline
def compare_interpolations(nodes):
    x = np.linspace(nodes[0, 0], nodes[-1, 0], 1000)

    # Obliczenia współczynników funkcji sklejanej sześciennej
    A, B, C = natural_cubic_spline(nodes)

    # Obliczenia wartości funkcji sklejanej sześciennej
    S_custom = cubic_spline_interpolation(x, nodes, A, B, C)

    # Utworzenie funkcji sklejanej sześciennej za pomocą CubicSpline
    cs = CubicSpline(nodes[:, 0], nodes[:, 1], bc_type='natural')

    # Obliczenia wartości funkcji CubicSpline
    S_cubic_spline = cs(x)

    # Rysowanie wykresu porównawczego
    plt.plot(nodes[:, 0], nodes[:, 1], 'ro', label='Data Points')
    plt.plot(x, S_custom, label='Custom Cubic Spline')
    plt.plot(x, S_cubic_spline, label='CubicSpline from scipy')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('Interpolated Values')
    plt.title('Comparison of Cubic Splines')
    plt.grid(True)
    plt.show()

# Porównanie custom cubic spline i CubicSpline
compare_interpolations(data)
