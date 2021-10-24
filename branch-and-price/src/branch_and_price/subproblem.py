import math
from typing import Tuple, Optional, List

import gurobipy as grb
from bidict import bidict

from common import TMachineSchedule, is_non_zero


class Subproblem:

    def __init__(self,
                 machine_id: int,
                 model: grb.Model,
                 task_to_variable: bidict[int, grb.Var]):

        self.machine_id = machine_id
        self._model = model
        self.task_to_variable = task_to_variable
        self._objective_value = None

    def solve(self):
        self._model.optimize()
        self._objective_value = self._model.getAttr(grb.GRB.Attr.ObjVal) \
            if self._model.status == grb.GRB.Status.OPTIMAL \
            else None

    def objective_value(self) -> Optional[float]:
        return self._objective_value

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

    def all_solutions(self) -> List[TMachineSchedule]:
        machine_schedules = []
        if self._model.status != grb.GRB.Status.OPTIMAL:
            return machine_schedules

        for k in range(self._model.solCount):
            self._model.Params.solutionNumber = k
            curr_obj_val = self._model.poolObjVal
            # if curr_obj_val < self._objective_value:
            #     continue
            machine_schedules.append(self._get_solution())
        return machine_schedules

    def _get_solution(self) -> TMachineSchedule:
        machine_schedule = None
        if self._model.status == grb.GRB.Status.OPTIMAL:

            tasks = [
                self.task_to_variable.inverse.get(var)
                for var in self._model.getVars()
                if is_non_zero(var.xn)
            ]

            machine_schedule = (self.machine_id, tasks)
        return machine_schedule
