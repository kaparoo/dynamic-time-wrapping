# -*- coding: utf-8 -*-


from typing import Callable
from typing import Final
from typing import List
from typing import Sequence
from typing import Union

# Make alias of type annotations for readability
Number = Union[int, float]
Matrix = List[List[Number]]
Function = Callable[[Number, Number], Number]


def display_matrix(x: Sequence[Number],
                   y: Sequence[Number],
                   D: Matrix):
    """Displays given matrix `D`."""

    N: Final = len(x)
    M: Final = len(y)

    for n in range(-1, N):
        for m in range(-1, M):
            if n > -1 and m > -1:
                print(f'[{D[n][m]}]', end=' ')
            elif (n, m) == (-1, -1):
                print(f'x\\y', end=' ')
            elif m == -1:
                print(f'[{x[n]}]', end=' ')
            elif n == -1:
                print(f'[{y[m]}]', end=' ')
        print()  # new line


def compute_accumulated_cost_matrix(x: Sequence[Number],
                                    y: Sequence[Number],
                                    c: Function = lambda a, b: abs(a-b)) -> Matrix:
    """Computes `accumulated cost matrix` from feature sequences `x` and `y`.

    Args:
        x: First feature sequence of size `N`.
        y: Second feature sequence of size `M`.
        c: Local cost measuring fuction. A function for euclidean distance
            is set to default.

    Returns:
        D: Accumulated cost matrix (`N` x `M`).
    """

    N: Final = len(x)
    M: Final = len(y)

    D: Matrix = [[0 for _ in range(M)] for _ in range(N)]

    for n in range(0, N):
        for m in range(0, M):
            if n == 0 and m == 0:
                D[0][0] = c(x[n], y[m])
            elif n == 0:  # m ¡ô [1, M-1]
                D[0][m] = c(x[n], y[m]) + D[0][m-1]
            elif m == 0:  # n ¡ô [1, N-1]
                D[n][0] = c(x[n], y[m]) + D[n-1][0]
            else:
                D[n][m] = c(x[n], y[m]) + \
                    min(D[n-1][m], D[n][m-1], D[n-1][m-1])

    return D


def classical_dtw(D: Matrix):
    """Operates (classical) dynamic time wrapping.

    Args:
        D: Accumulated cost matrix of size N x M.

    Returns:
        p: Optimal alignment path.
    """

    N: Final[Number] = len(D)
    M: Final[Number] = len(D[0])

    p = []
    (n, m) = (N-1, M-1)
    while (n, m) != (0, 0):
        p.append((n, m))  # Insert current point to the path

        # Assume (n, m) ¡ô [0, N-1] ¡¿ [0, M-1]
        candidates = []
        if n == 0:  # m > 0
            candidates.append((0, m-1))
        elif m == 0:  # n > 0
            candidates.append((n-1, 0))
        else:  # n > 0 and m > 0
            candidates.append((n-1, m-1))
            if D[n-1][m] != D[n-1][m-1]:
                candidates.append((n-1, m))
            if D[n][m-1] != D[n-1][m-1]:
                candidates.append((n, m-1))

        # Find next point that requires least cost to move
        (n, m) = min(candidates, key=lambda point: D[point[0]][point[1]])
    else:
        p.append((0, 0))

    return p[::-1]


if __name__ == '__main__':
    x = [1, 2, 3, 3, 2, 1]
    y = [1, 1, 2, 3, 3, 2]
    D = compute_accumulated_cost_matrix(x, y)
    display_matrix(x, y, D)
    p = classical_dtw(D)

    print(f'Path: {p}')
