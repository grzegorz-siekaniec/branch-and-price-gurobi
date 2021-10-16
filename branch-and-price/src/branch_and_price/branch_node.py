import copy
import itertools
import logging
from collections import defaultdict
from typing import List, Dict, Optional

import gurobipy as grb

from branch_and_price.branching_rule import BranchingRule
from branch_and_price.subproblem_builder import SubproblemBuilder
from common import TMachineSchedule, is_integer, TAssignment, is_non_zero
from input_data import GeneralAssignmentProblem


class BranchNode:

    next_node_id = itertools.count(start=0)

    def __init__(self,
                 gap_instance: GeneralAssignmentProblem,
                 branching_rules: List[BranchingRule],
                 machine_schedules: List[TMachineSchedule]):

        self._id = next(self.next_node_id)
        self.machine_schedules = self._filter_machine_schedule_based_on_branching_rule(
            branching_rules,
            machine_schedules
        )

        self.branching_rules = branching_rules
        self.gap_instance = gap_instance

        self._rmp = grb.Model('GAP_RMP')

        self.next_machine_schedule_index = itertools.count(start=0)
        self.machine_schedule_index = dict()
        self.machine_schedule_index_to_variable = dict()
        self.machine_to_assignment_constraint: Dict = dict()
        self.task_to_assignment_constraint: Dict = dict()

    def is_feasible(self) -> bool:
        """Returns true is solution is optimal"""
        status = self._rmp.getAttr(grb.GRB.Attr.Status)
        feasible_statuses = {grb.GRB.Status.OPTIMAL, grb.GRB.Status.SUBOPTIMAL}
        return status in feasible_statuses

    def has_integer_solution(self) -> bool:
        """
        Return true if RMP solution is integer.
        """
        ret = all(is_integer(var.x) for var in self._rmp.getVars())
        return ret

    def machine_task_to_branch_on(self) -> Optional[TAssignment]:
        """
        Identifies machine and task that algorithm can further branch on.
        To do so it checks all base columns and identifies assignment
        of task to machine that is non-integer. However one needs to be careful
        as columns represents different solutions to knapsack problem. So there
        could be to columns each with value 0.5 that assign task `t` to machine `m`.
        In this case assignment of `t` to `m` would be integer and such tuple
        will not be returned.
        :return: Return machine_id, task_id
        """
        machine_task_to_value = defaultdict(lambda: 0)

        for var in self._rmp.getVars():
            if is_non_zero(var.x):
                assignment_idx = self.machine_schedule_index_to_variable.inverse.get(var)
                machine_schedule = self.machine_schedule_index[assignment_idx]
                machine = machine_schedule[0]
                tasks = machine_schedule[1]
                for task in tasks:
                    machine_task_to_value[(machine, task)] += var.x

        for (machine, task), val in machine_task_to_value.items():
            if not is_integer(val):
                return machine, task

        return None

    def solve(self):
        self._solve_using_column_generation()

    def _solve_using_column_generation(self):
        logging.info("[CG] Solving GAP using column generation")

        self._rmp.setAttr(grb.GRB.Attr.ModelSense, grb.GRB.MAXIMIZE)

        self._build_constraints()

        self._filter_columns_based_on_branching_rule()
        self._add_initial_columns()

        optimal = False
        itr_cnt = itertools.count(start=1)
        subproblem_builder = SubproblemBuilder(gap_instance=self.gap_instance)

        while not optimal:
            col_gen_itr = next(itr_cnt)
            logging.debug("[CG] Column generation iteration ... {}".format(col_gen_itr))

            self._rmp.optimize()

            logging.debug("[CG]  * Objective value: {:.2f}".format(self._rmp.getObjective().getValue()))

            # obtain duals associated with tasks, solution might be infeasible
            # but duals will be returned
            task_duals = self._rmp.solution.get_dual_values(self.task_bind_row_idx.values())
            machine_duals = self._rmp.solution.get_dual_values(self.machine_bind_row_id.values())

            optimal = True
            for machine_id in range(self.gap_instance.num_machines):
                logging.debug("[CG]  * Solving subproblem for machine {}".format(machine_id))

                machine_dual = machine_duals[machine_id]

                subproblem = subproblem_builder.build(machine_id=machine_id,
                                                      machine_dual=machine_dual,
                                                      task_duals=task_duals,
                                                      branching_rules=self.branching_rules)

                subproblem.solve()
                subproblem_objective_value = subproblem.objective_value()
                if subproblem_objective_value and subproblem_objective_value > 0:
                    optimal = False
                    subproblem_solution = subproblem.solution()
                    self._add_column_to_rmp(subproblem_solution)

    def _build_constraints(self):
        self._build_task_binding_constraints()
        self._build_machine_binding_constraints()

    def _build_task_binding_constraints(self):

        for task_id in range(self.gap_instance.num_tasks):
            lhs = grb.quicksum([])
            rhs = 1
            name = f'task_assignment_{task_id}'
            c = self._rmp.addConstr(lhs <= rhs, name=name)
            self.task_to_assignment_constraint[task_id] = c

    def _build_machine_binding_constraints(self):
        for machine_id in range(self.gap_instance.num_machines):
            lhs = grb.quicksum([])
            rhs = 1
            name = f'machine_assignment_{machine_id}'
            c = self._rmp.addConstr(lhs <= rhs, name=name)
            self.machine_to_assignment_constraint[machine_id] = c

    def _add_initial_columns(self):
        """
        Adds initial columns which are base columns of all parent nodes.
        """

        for machine_schedule in self.machine_schedules:
            self._add_column_to_rmp(machine_schedule)

    def _add_column_to_rmp(self, machine_schedule: TMachineSchedule):
        """
        Adds a column that represents assigning `tasks` to `machine_id` to RMP.
        It does so by performing two steps:
        (1) creating variable and adding to Gurobi problem/model.
        (2) storing information about what kind of assignment a column represents.
        :param machine_id: machine
        :param tasks: tasks assigned to machine
        """
        machine_schedule_index = next(self.next_machine_schedule_index)
        self.machine_schedule_index[machine_schedule_index] = copy.deepcopy(machine_schedule)

        machine_id = machine_schedule[0]
        tasks = machine_schedule[1]

        profit = self.gap_instance.machine_schedule_profit(machine_schedule)
        name = None #"y_{}_{}".format(machine_id, k)

        c = grb.Column()
        coeff = 1.0
        for task in tasks:
            constr = self.task_to_assignment_constraint[task]
            c.addTerms(coeff, constr)

        machine_constr = self.machine_to_assignment_constraint[machine_id]
        c.addTerms(coeff, machine_constr)

        var = self._rmp.addVar(
            lb=0.0,
            ub=1.0,
            obj=profit,
            vtype=grb.GRB.CONTINUOUS,
            name=name,
            column=c
        )

        self.machine_schedule_index_to_variable[machine_schedule_index] = var

    @classmethod
    def _filter_machine_schedule_based_on_branching_rule(cls,
                                                         branching_rules: List[BranchingRule],
                                                         machine_schedules: List[TMachineSchedule])\
            -> List[TMachineSchedule]:
        """
        Filters machine schedules passed from parent nodes according to
        defined branching rules where each each schedule represents assignments of
        tasks to a machine. Column is filtered out:
        (1) If branching rule forces task to be assigned to a machine:
            (i) if column represents assignments of tasks to a different machine
                and task is in the list of assigned task.
            (ii) if column represents assignments of tasks to a machine
                but task is missing in the list of assigned tasks.
        (2) If branching rule forbids assigning a task to machine, and column
            represents assignment such that a task is assigned to a machine.
        """
        if len(branching_rules) == 0:
            return []

        tmp_cols = []
        for machine, tasks in machine_schedules:
            legal = True
            for br in branching_rules:
                if br.assigned is True and br.machine == machine and br.task not in tasks:
                    legal = False
                    break
                if br.assigned is True and br.machine != machine and br.task in tasks:
                    legal = False
                    break
                if br.assigned is False and br.machine == machine and br.task in tasks:
                    legal = False
                    break

            if legal is True:
                tmp_cols.append((machine, tasks))

        return tmp_cols
