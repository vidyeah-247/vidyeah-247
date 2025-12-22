import numpy as np


def gaussian(A, b):
    A = np.array(A, float)
    b = np.array(b, float)
    n = len(b)

    Ab = np.hstack((A, b.reshape(-1, 1)))

    # forward elimination
    for i in range(n):
        for k in range(i+1, n):
            m = Ab[k, i] / Ab[i, i]
            Ab[k] = Ab[k] - m * Ab[i]

    # back substitution
    X = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = Ab[i, -1]
        for j in range(i+1, n):
            s -= Ab[i, j] * X[j]
        X[i] = s / Ab[i, i]

    return X


def gauss_jordan(A, b):
    A = np.array(A, float)
    b = np.array(b, float).reshape(-1, 1)
    Ab = np.hstack((A, b))
    n = len(A)

    for i in range(n):
        Ab[i] = Ab[i] / Ab[i, i]
        for j in range(n):
            if j != i:
                Ab[j] -= Ab[j, i] * Ab[i]

    return Ab[:, -1]


def jacobi(A, b, tol=1e-6, max_iter=100):
    A = np.array(A, float)
    b = np.array(b, float)
    n = len(A)
    x = np.zeros(n)

    for _ in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]

        if np.linalg.norm(x_new - x) < tol:
            return x_new

        x = x_new

    return x


def gauss_seidel(A, b, tol=1e-6, max_iter=100):
    A = np.array(A, float)
    b = np.array(b, float)
    n = len(A)
    x = np.zeros(n)

    for _ in range(max_iter):
        x_old = np.copy(x)

        for i in range(n):
            s1 = sum(A[i][j] * x[j] for j in range(i))
            s2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))

            x[i] = (b[i] - (s1 + s2)) / A[i][i]

        if np.linalg.norm(x - x_old) < tol:
            return x

    return x
