from itertools import product
from typing import Dict, Tuple

from bidict import bidict
import gurobipy as grb

from input_data import GeneralAssignmentProblem
from standalone_model.gap_standalone_model import GAPStandaloneModel


class GAPStandaloneModelBuilder:

    def __init__(self, gap_instance: GeneralAssignmentProblem):

        self._gap_instance = gap_instance

        self.model = grb.Model("gap_standalone_model")
        self.model.setAttr(grb.GRB.Attr.ModelSense, grb.GRB.MAXIMIZE)

        self.task_machine_to_variable: Dict[Tuple[int, int], grb.Var] = dict()

    def build(self) -> GAPStandaloneModel:
        self._build_columns()
        self._build_constraints()
        self.model.update()

        return GAPStandaloneModel(
            self.model,
            bidict(self.task_machine_to_variable)
        )

    def _build_columns(self):
        tasks_times_machines = product(
            range(self._gap_instance.num_tasks),
            range(self._gap_instance.num_machines))

        # change to addVars
        for task_id, machine_id in tasks_times_machines:
            assignment = (task_id, machine_id)
            name = f'task_{task_id}_machine_{machine_id}'
            profit = self._gap_instance.assignment_profit(task_id, machine_id)
            var = self.model.addVar(
                lb=0.0,
                ub=1.0,
                obj=profit,
                vtype=grb.GRB.BINARY,
                name=name,
            )
            self.task_machine_to_variable[assignment] = var

    def _build_constraints(self):
        self._build_assignment_constraints()
        self._build_capacity_constraints()

    def _build_assignment_constraints(self):
        for task_id in range(self._gap_instance.num_tasks):
            lhs = grb.quicksum([
                self.task_machine_to_variable[(task_id, machine_id)]
                for machine_id in range(self._gap_instance.num_machines)
            ])
            rhs = 1
            name = f'task_assignment_{task_id}'
            self.model.addConstr(lhs <= rhs, name=name)

    def _build_capacity_constraints(self):
        for machine_id in range(self._gap_instance.num_machines):
            lhs = grb.quicksum([
                self._gap_instance.weight(task_id, machine_id) * self.task_machine_to_variable[(task_id, machine_id)]
                for task_id in range(self._gap_instance.num_tasks)
            ])
            rhs = self._gap_instance.machine_capacity(machine_id)
            name = f'machine_capacity_{machine_id}'
            self.model.addConstr(lhs <= rhs, name=name)
