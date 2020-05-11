import numpy as np


# Helper module

def arithmetic_progression_series(a, d, n):
    curr_term = a
    series = [a]
    for i in range(1, n):
        next_term = curr_term + d
        series.append(np.round(next_term, 2))
        curr_term = next_term
    return series


if __name__ == '__main__':
    print(arithmetic_progression_series(0.3, 0.1, 10))
