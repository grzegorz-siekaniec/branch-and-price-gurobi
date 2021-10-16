from typing import Dict, List

import gurobipy as grb
from bidict import bidict

from common import TMachineSchedule
from input_data import GeneralAssignmentProblem
from standalone_model import FeasibleMachineSchedulesFinder
from standalone_model.dantzig_wolfe_formulation_gap_standalone_model import DantzigWolfeFormulationGapStandaloneModel


class DantzigWolfeFormulationGapStandaloneModelBuilder:

    def __init__(self, gap_instance: GeneralAssignmentProblem):
        self._gap_instance = gap_instance
        self.machine_schedule_idx_to_variable: Dict[int, grb.Var] = dict()
        self.feasible_machine_schedules: List[TMachineSchedule] = list()

        self.dw_model = grb.Model("dantzig_wolfe_formulation_gap_standalone_model")
        self.dw_model.setAttr(grb.GRB.Attr.ModelSense, grb.GRB.MAXIMIZE)

    def build(self) -> DantzigWolfeFormulationGapStandaloneModel:
        self.feasible_machine_schedules = FeasibleMachineSchedulesFinder(self._gap_instance).find()

        self._build_columns()
        self._build_convexity_constraints()
        self._build_assignment_constraints()

        self.dw_model.update()

        return DantzigWolfeFormulationGapStandaloneModel(
            dw_model=self.dw_model,
            feasible_machine_schedules=self.feasible_machine_schedules,
            machine_schedule_idx_to_variable=bidict(self.machine_schedule_idx_to_variable)
        )

    def _build_columns(self):
        for idx, machine_schedule in enumerate(self.feasible_machine_schedules):
            machine_id = machine_schedule[0]
            tasks = machine_schedule[1]
            profit = self._gap_instance.machine_schedule_profit(machine_schedule)
            name = f"machine_{machine_id}_tasks_{'_'.join(str(task_id) for task_id in tasks)}"
            var = self.dw_model.addVar(
                lb=0.0,
                ub=1.0,
                obj=profit,
                vtype=grb.GRB.BINARY,
                name=name,
            )

            self.machine_schedule_idx_to_variable[idx] = var

    def _build_convexity_constraints(self):
        for machine_id in range(self._gap_instance.num_machines):
            lhs = grb.quicksum([
                1 * self.machine_schedule_idx_to_variable[idx]
                for idx, machine_schedule in enumerate(self.feasible_machine_schedules)
                if machine_schedule[0] == machine_id
            ])
            rhs = 1
            name = f'convexity_{machine_id}'
            self.dw_model.addConstr(lhs == rhs, name=name)

    def _build_assignment_constraints(self):
        for task_id in range(self._gap_instance.num_tasks):
            lhs = grb.quicksum([
                1 * self.machine_schedule_idx_to_variable[idx]
                for idx, machine_schedule in enumerate(self.feasible_machine_schedules)
                if task_id in machine_schedule[1]
            ])
            rhs = 1
            name = f'task_assignment_{task_id}'
            self.dw_model.addConstr(lhs == rhs, name=name)
