"""
File contains implementation of full model of GAP
and it is expected to serve as a verification that
branch and price returns the correct results.
"""
import logging
from collections import defaultdict
from typing import Tuple, Dict, Collection

import gurobipy as grb

from common import is_non_zero
from input_data import GeneralAssignmentProblem
from bidict import bidict


class GAPStandaloneModel:

    def __init__(self,
                 model: grb.Model,
                 assignment_to_variable: bidict[Tuple[int, int], grb.Var]):
        self.mip_model = model
        self._assignment_to_variable = assignment_to_variable

    def solve(self):
        self.mip_model.Params.LogToConsole = 0
        self.mip_model.optimize()

    def write(self):
        model_name = self.mip_model.getAttr(grb.GRB.Attr.ModelName)
        self.mip_model.write(f'{model_name}.lp')

    def report_results(self):
        obj_val = self.mip_model.getAttr(grb.GRB.Attr.ObjVal)

        logging.info("** Final results using standalone model! **")
        logging.info("Objective value: %f", obj_val)

        machine_to_tasks: Dict[int, Collection[int]] = defaultdict(set)
        for var in self.mip_model.getVars():
            if is_non_zero(var.x):
                assignment = self._assignment_to_variable.inverse.get(var)
                task_id = assignment[0]
                machine_id = assignment[1]
                machine_to_tasks[machine_id].add(task_id)

        logging.info("Machine -> Set of tasks")

        for machine in sorted(machine_to_tasks):
            tasks = machine_to_tasks[machine]
            logging.info(f'{machine}\t{" ".join([str(task) for task in tasks])}')

        logging.info('')
