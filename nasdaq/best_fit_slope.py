from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random as r

DATA_POINTS = 30
style.use('fivethirtyeight')


def create_y_list(y_list, current, total, val, variance, step, correlation):
    if current < total:
        y = val + r.randrange(-variance, variance)
        y_list.append(y)
        if correlation == 'pos':
            val += step
        elif correlation == 'neg':
            val -= step

        current += 1
        return create_y_list(y_list, current, total, val, variance, step, correlation)
    else:
        return y_list


def create_dataset(total, variance, step=1, correlation='pos'):
    y_list = create_y_list(y_list=[], current=0, total=total, val=1, variance=variance, step=step, correlation=correlation)
    x_list = [x for x in range(total)]
    return np.array(x_list, dtype=np.float64), np.array(y_list, dtype=np.float64)


def compute_best_fit_slope(x_list, y_list):
    numerator = (mean(x_list) * mean(y_list)) - mean(x_list * y_list)
    denominator = mean(x_list) ** 2 - mean(x_list ** 2)
    return numerator / denominator


def compute_y_intercept(y_list, x_list, m):
    return mean(y_list) - (m * mean(x_list))


def compute_squared_error(y_mean_line, y_org_list):
    return sum((y_mean_line - y_org_list) ** 2)


def compute_r_squared(y_list_original, regression_line):
    y_mean_line = [mean(y_list_original) for _ in y_list_original]
    sq_error_y_mean_line = compute_squared_error(y_mean_line, y_list_original)
    sq_error_regression_line = compute_squared_error(regression_line, y_list_original)
    return 1 - (sq_error_regression_line / sq_error_y_mean_line)


def create_regression_line(x_list, m, c):
    return np.array([(m*x) + c for x in x_list], dtype=np.float64)


def predict(x, m, b):
    y = m * x + b
    plt.scatter(x, y, color='r')


def graph(scatter={}, plot={}):
    if 'x' in scatter and 'y' in scatter:
        plt.scatter(scatter['x'], scatter['y'])

    if 'x' in plot and 'y' in plot:
        plt.plot(plot['x'], plot['y'])

    plt.show()


if __name__ == '__main__':
    x_list, y_list = create_dataset(total=DATA_POINTS, variance=20, step=2, correlation='pos')

    m = compute_best_fit_slope(x_list, y_list)
    print(f'best fit slope: {m}')

    c = compute_y_intercept(y_list, x_list, m)
    print(f'y-intercept: {c}')

    regression_line = create_regression_line(x_list, m, c)

    r_squared = compute_r_squared(y_list, regression_line)
    print(f'r_squared: {r_squared}')

    predict(x=45, m=m, b=c)
    graph(
        scatter={'x': x_list,  'y': y_list},
        plot={'x': x_list, 'y': regression_line}
    )
