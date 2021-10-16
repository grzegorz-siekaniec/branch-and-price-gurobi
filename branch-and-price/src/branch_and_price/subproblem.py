from typing import Tuple, Optional

import gurobipy as grb
from bidict import bidict

from common import TMachineSchedule, is_non_zero


class Subproblem:

    def __init__(self,
                 machine_id: int,
                 model: grb.Model,
                 task_to_variable: bidict[Tuple[int, int]]):

        self.machine_id = machine_id
        self._model = model
        self.task_to_variable = task_to_variable

    def solve(self):
        self._model.optimize()

    def objective_value(self) -> Optional[float]:
        return \
            self._model.getAttr(grb.GRB.Attr.ObjVal) \
            if self._model.status == grb.GRB.Status.OPTIMAL \
            else None

    def solution(self) -> Optional[TMachineSchedule]:
        machine_schedule = None
        if self._model.status == grb.GRB.Status.OPTIMAL:

            tasks = [
                self.task_to_variable.inverse.get(var)
                for var in self._model.getVars()
                if is_non_zero(var.x)
            ]

            machine_schedule = (self.machine_id, tasks)
        return machine_schedule
