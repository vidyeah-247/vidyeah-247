from flask import Flask, render_template, request, jsonify
import sympy as sp
from methods.root_methods import *
from methods.linear_methods import *
import os

app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/solve_root", methods=["POST"])
def solve_root():
    data = request.json
    method = data["method"]
    f_str = data["function"]

    x = sp.symbols("x")
    f_expr = sp.sympify(f_str)
    f = sp.lambdify(x, f_expr, "math")

    if method == "bisection":
        root = bisection(f, data["a"], data["b"])
    elif method == "secant":
        root = secant(f, data["a"], data["b"])
    elif method == "false_position":
        root = false_position(f, data["a"], data["b"])
    elif method == "newton":
        root = newton_raphson(f_expr, data["a"])
    else:
        return jsonify({"error": "Invalid method"}), 400

    return jsonify({"root": float(root)})


@app.route("/solve_linear", methods=["POST"])
def solve_linear():
    data = request.json
    method = data["method"]
    A = data["A"]
    b = data["b"]

    if method == "gaussian":
        x = gaussian(A, b)
    elif method == "gauss_jordan":
        x = gauss_jordan(A, b)
    elif method == "jacobi":
        x = jacobi(A, b)
    elif method == "gauss_seidel":
        x = gauss_seidel(A, b)
    else:
        return jsonify({"error": "Invalid method"}), 400

    return jsonify({"solution": x.tolist()})


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
