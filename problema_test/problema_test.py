# Importing numerical libraries
import numpy as np
from scipy import optimize

# Importing plotting libraries
from bokeh.plotting import figure, show
from bokeh.io import export_png

export = False

def solve_NLP(N):

    # Defining problem bound conditions

    a = 0
    b = 1

    x_start = 0
    x_end = 1
    v_start = 0
    v_end = 0

    t = np.linspace(a, b, num=N)

    h = t[1:N] - t[0:N-1]

    # Defining the objective function (using trapezoidal quadrature)

    def objective_function(X):
        return 0.5 * np.dot(h, (np.power(X[2*N:3*N-1], 2) + np.power(X[2*N+1:3*N], 2)))

    # Setting bound conditions

    lb = np.full(3*N, -np.inf, dtype=float)
    ub = np.full(3*N, +np.inf, dtype=float)

    lb[0] = x_start
    ub[0] = x_start
    lb[N-1] = x_end
    ub[N-1] = x_end
    lb[N] = v_start
    ub[N] = v_start
    lb[2*N-1] = v_end
    ub[2*N-1] = v_end

    bounds = optimize.Bounds(lb, ub)

    # Setting dynamics constraints

    constraints = []

    for i in range(N-1):
        def fun_position(X, i=i): return X[i+1] - \
            X[i] - 0.5 * h[i] * (X[N+i+1] + X[N+i])
        def fun_velocity(X, i=i): return X[N+i+1] - \
            X[N+i] - 0.5 * h[i] * (X[2*N+i+1] + X[2*N+i])
        const_position = {"fun": fun_position, "type": "eq"}
        const_velocity = {"fun": fun_velocity, "type": "eq"}
        constraints.append(const_position)
        constraints.append(const_velocity)

    # Setting an initial guess

    initial_guess = np.zeros(3*N)
    initial_guess[0:N] = np.linspace(x_start, x_end, num=N)
    initial_guess[N:2*N] = 1

    # Utility function called at the end of each iteration of the optimizer

    def callback(a):
        print()

    # Solving the optimization problem

    res = optimize.minimize(
        objective_function,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    # Returning optimization results

    x = res.x[0:N]
    v = res.x[N:2*N]
    u = res.x[2*N:3*N]

    print(res.message)

    return t, x, v, u


if __name__ == '__main__':

    t, x, v, u = solve_NLP(20)

    z = np.linspace(0, 1, num=1000)

    # Plotting position s(t)
    p = figure(x_axis_label='Time t', y_axis_label='Position x(t)')
    p.toolbar_location = None
    p.line(z, 3*z**2 - 2*z**3, line_width=2, line_color="orange")
    p.circle(t, x, line_width=2)
    if export:
        export_png(p, filename="problema_test_position.png", width=400, height=250)
    else:
        show(p)

    # Plotting velocity v(t)
    p = figure(x_axis_label='Time t', y_axis_label='Velocity v(t)')
    p.toolbar_location = None
    p.line(z, 6*z - 6*z**2, line_width=2, line_color="orange")
    p.circle(t, v, line_width=2)
    if export:
        export_png(p, filename="problema_test_velocity.png", width=400, height=250)
    else:
        show(p)

    #   Plotting control function u(t)
    p = figure(x_axis_label='Time t', y_axis_label='Control u(t)')
    p.toolbar_location = None
    p.line(t, 6-12*t, line_width=2, line_color="orange")
    p.circle(t, u, line_width=2)
    if export:
        export_png(p, filename="problema_test_control.png", width=400, height=250)
    else:
        show(p)