from typing import Tuple, List
from math import isclose

import numpy as np
import gurobipy.gurobipy as grb

# change to use dataclass?
TAssignment = Tuple[int, int]  # machine -> task

# change list to frozenset?
TTask = int
TMachineSchedule = Tuple[int, List[int]]  # machine -> tasks
TCompleteSchedule = List[TMachineSchedule]


def is_non_zero(var):
    return np.abs(var) > 0.0001


def is_integer(bool_var_val: float):
    """Return true is variable that is supposed to represent binary variable
    is integer. More specifically if it differs from integer value by less than 1e-05."""
    eps = 1e-05
    return isclose(bool_var_val, 0.0, abs_tol=eps) or isclose(bool_var_val, 1.0, abs_tol=eps)


def has_solution(model_status) -> bool:
    """
    Uses model status to determine whether model has
    a solution and asking for objective value make sense
    :param model_status: model status
    :return: True if model has solution
    """
    return model_status in {grb.GRB.Status.OPTIMAL, grb.GRB.Status.SUBOPTIMAL}
