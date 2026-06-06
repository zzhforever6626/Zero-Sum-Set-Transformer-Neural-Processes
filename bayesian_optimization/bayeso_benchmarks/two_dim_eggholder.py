import numpy as np

from bayeso_benchmarks.benchmark_base import Function


def fun_target(bx, dim_bx):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx

    x = bx[0]
    y = bx[1]

    # Standard 2D EggHolder function (minimization benchmark)
    # f(x, y) = -(y + 47) * sin(sqrt(|x/2 + y + 47|)) - x * sin(sqrt(|x - (y + 47)|))
    term1 = -(y + 47.0) * np.sin(np.sqrt(np.abs(x / 2.0 + y + 47.0)))
    term2 = -x * np.sin(np.sqrt(np.abs(x - (y + 47.0))))

    return float(term1 + term2)


class Eggholder(Function):
    def __init__(self):
        dim_bx = 2
        bounds = np.array([
            [-5.12, 5.12],
            [-5.12, 5.12],
        ])

        # Known global minimizer (commonly cited)
        global_minimizers = np.array([
            [5.12, 4.042319],
        ])

        # Make sure it matches the above minimizer (use consistent precision)
        global_minimum = float(fun_target(global_minimizers[0], dim_bx))

        function = lambda bx: fun_target(bx, dim_bx)
        Function.__init__(self, dim_bx, bounds, global_minimizers, global_minimum, function)
