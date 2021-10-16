import logging
from typing import List, Set, Dict

import gurobipy as grb
from bidict import bidict

from common import TMachineSchedule, is_non_zero


class DantzigWolfeFormulationGapStandaloneModel:

    def __init__(self,
                 dw_model: grb.Model,
                 machine_schedule_idx_to_variable: bidict[int, grb.Var],
                 feasible_machine_schedules: List[TMachineSchedule]):
        self.dw_model = dw_model
        self.feasible_machine_schedules = feasible_machine_schedules
        self.machine_schedule_idx_to_variable = machine_schedule_idx_to_variable

    def solve(self):
        self.dw_model.Params.LogToConsole = 0
        self.dw_model.optimize()

    def write(self):
        model_name = self.dw_model.getAttr(grb.GRB.Attr.ModelName)
        self.dw_model.write(f'{model_name}.lp')

    def report_results(self):
        obj_val = self.dw_model.getAttr(grb.GRB.Attr.ObjVal)

        logging.info("** Final results using Dantzig-Wolfe formulation of standalone model! **")
        logging.info("Objective value: %f", obj_val)

        machine_to_tasks: Dict[int, Set[int]] = dict()
        for var in self.dw_model.getVars():
            if is_non_zero(var.x):
                machine_schedule = self._get_machine_schedule(var)
                machine_id = machine_schedule[0]
                tasks = machine_schedule[1]

                machine_to_tasks[machine_id] = set(tasks)

        logging.info("Machine -> Set of tasks")
        for machine in sorted(machine_to_tasks):
            tasks = machine_to_tasks[machine]
            logging.info(f'{machine}\t{" ".join([str(task) for task in tasks])}')

        logging.info('')

    def _get_machine_schedule(self, var: grb.Var) -> TMachineSchedule:
        idx = self.machine_schedule_idx_to_variable.inverse.get(var)
        return self.feasible_machine_schedules[idx]