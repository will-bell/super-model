import cvxpy
import numpy as np


def qp_min_effort(Aclf: np.ndarray, bclf: float, Acbf: np.ndarray, bcbf: np.ndarray, m: float) -> np.ndarray:
    u = cvxpy.Variable((2, 1))
    delta = cvxpy.Variable((1, 1))

    constraints = [Aclf @ u + bclf <= delta]
    if Acbf is not None:
        constraints.append(Acbf @ u + bcbf <= 0)

    problem = cvxpy.Problem(
        cvxpy.Minimize(cvxpy.norm2(u)**2 + m*delta**2),
        constraints)
    problem.solve(solver=cvxpy.MOSEK)
    u_star = np.transpose(u.value)

    return u_star
