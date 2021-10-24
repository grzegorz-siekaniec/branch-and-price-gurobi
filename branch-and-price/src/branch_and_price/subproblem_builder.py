from typing import Tuple, List, Dict
import gurobipy as grb
from bidict import bidict

from branch_and_price.branching_rule import BranchingRule
from branch_and_price.subproblem import Subproblem
from common import TAssignment
from input_data import GeneralAssignmentProblem


class SubproblemBuilder:

    def __init__(self,
                 gap_instance: GeneralAssignmentProblem):
        self._gap_instance = gap_instance

    def build(self,
              machine_id: int,
              machine_dual: float,
              task_duals: List[float],
              branching_rules: List[BranchingRule]):

        model = grb.Model('GAP_Subproblem')
        model.setAttr(grb.GRB.Attr.ModelSense, grb.GRB.MAXIMIZE)
        model.setAttr(grb.GRB.Attr.ObjCon, -machine_dual)
        model.Params.LogToConsole = 0

        task_to_variable = self._build_columns(
            model,
            machine_id,
            task_duals,
            branching_rules
        )

        self._build_capacity_constraint(
            model,
            machine_id,
            task_to_variable
        )

        model.update()

        return Subproblem(
            machine_id=machine_id,
            model=model,
            task_to_variable=bidict(task_to_variable))

    def _build_capacity_constraint(self,
                                   model: grb.Model,
                                   machine_id: int,
                                   task_to_variable: Dict[int, grb.Var]):
        lhs = grb.quicksum([
            self._gap_instance.weight(task_id, machine_id) * task_to_variable[task_id]
            for task_id in range(self._gap_instance.num_tasks)
        ])
        rhs = self._gap_instance.machine_capacity(machine_id)
        name = f'machine_capacity_{machine_id}'
        model.addConstr(lhs <= rhs, name=name)

    def _build_columns(self, model, machine_id, task_duals, branching_rules) -> Dict[int, grb.Var]:
        task_to_variable: Dict[int, grb.Var] = dict()

        for task_id in range(self._gap_instance.num_tasks):
            lb, ub = self._lower_and_upper_bound(machine_id, task_id, branching_rules)
            name = f'task_{task_id}_machine_{machine_id}'
            obj = self._gap_instance.assignment_profit(task_id=task_id, machine_id=machine_id) - task_duals[task_id]
            var = model.addVar(
                lb=lb,
                ub=ub,
                obj=obj,
                vtype=grb.GRB.BINARY,
                name=name,
            )
            task_to_variable[task_id] = var

        return task_to_variable

    @classmethod
    def _lower_and_upper_bound(cls, machine: int, task: int, branching_rules: List[BranchingRule]) -> Tuple[float, float]:
        """
        Returns lower and upper bound for a task variable
        while solving knapsack problem for a machine by considering
        branching rule.
        (1) If branching rules force to assign task to a machine,
            then lower and upper bound are set to `1`.
        (2) If branching rules forbid to assign task to a machine,
            then lower and upper bound is `0`.
        (3) Otherwise, it is set to `0` and `1` respectively.
        """
        lb = 0.0
        ub = 1.0
        for br in branching_rules:
            if br.task == task:
                if br.machine == machine and br.assigned is True:
                    lb = 1
                    break
                elif br.machine == machine and br.assigned is False:
                    ub = 0
                    break
                elif br.machine != machine and br.assigned is True:
                    ub = 0
                    break
        return lb, ub
