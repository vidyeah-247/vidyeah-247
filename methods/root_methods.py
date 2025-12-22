import sympy as sp

def bisection(f, a, b, tol=1e-6, max_iter=100):
    if f(a) * f(b) > 0:
        raise ValueError("Root not bracketed")

    for _ in range(max_iter):
        c = (a + b) / 2
        if abs(f(c)) < tol:
            return c
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return c


def newton_raphson(f_expr, x0, tol=1e-6, max_iter=100):
    x = sp.symbols("x")
    df_expr = sp.diff(f_expr, x)

    f = sp.lambdify(x, f_expr, "math")
    df = sp.lambdify(x, df_expr, "math")

    for _ in range(max_iter):
        x1 = x0 - f(x0) / df(x0)
        if abs(x1 - x0) < tol:
            return x1
        x0 = x1

    return x1


def secant(f, x0, x1, tol=1e-6, max_iter=100):
    for _ in range(max_iter):
        fx0 = f(x0)
        fx1 = f(x1)

        if fx1 - fx0 == 0:
            raise ValueError("Division by zero")

        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)

        if abs(x2 - x1) < tol:
            return x2

        x0, x1 = x1, x2

    return x2


def false_position(f, a, b, tol=1e-6, max_iter=100):
    if f(a) * f(b) > 0:
        raise ValueError("Root not bracketed")

    for _ in range(max_iter):
        c = (a*f(b) - b*f(a)) / (f(b) - f(a))
        if abs(f(c)) < tol:
            return c
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return c




    
 
