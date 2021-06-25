import time
import problema_test

# Importing plotting libraries
from bokeh.plotting import figure, show
from bokeh.io import export_png

export = False

if __name__ == '__main__':

    n = []
    execution_time = []

    for i in range(1, 400, 10):
        start = time.time()
        t, x, v, u = problema_test.solve_NLP(i)
        end = time.time()
        n.append(i)
        execution_time.append(end-start)

    p = figure(title="Execution Time (s)", x_axis_label='# collocation points', y_axis_label='Execution Time (s)')
    p.toolbar_location = None
    p.circle(n, execution_time, line_width=2)

    if export:
        export_png(p, filename="problema_test_position.png", width=400, height=250)
    else:
        show(p)