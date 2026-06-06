import numpy as np

from bayeso_benchmarks.benchmark_base import Function


def fun_target(bx, dim_bx):
    assert len(bx.shape) == 1
    assert bx.shape[0] == dim_bx

    x, y = bx[0], bx[1]
    num = (np.sin(x**2 - y**2) ** 2) - 0.5
    den = (1.0 + 0.001 * (x**2 + y**2)) ** 2
    return 0.5 + (num / den)


class SchafferN2(Function):
    """Schaffer Function N.2 (2D), configured for minimization.

    - Bounds aligned with your DropWave setup: [-5.12, 5.12]^2
    - Global minimum: f(0, 0) = 0
    """
    def __init__(self):
        dim_bx = 2
        bounds = np.array([
            [-5.12, 5.12],
            [-5.12, 5.12],
        ])
        global_minimizers = np.array([
            [0.0, 0.0],
        ])
        global_minimum = 0.0

        function = lambda bx: fun_target(bx, dim_bx)
        Function.__init__(self, dim_bx, bounds, global_minimizers, global_minimum, function)
