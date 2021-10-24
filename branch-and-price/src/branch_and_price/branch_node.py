import copy
import itertools
import logging
import math
from collections import defaultdict
from typing import List, Dict, Optional

import gurobipy as grb
import numpy as np
import scipy.optimize
from bidict import bidict

from branch_and_price.branching_rule import BranchingRule
from branch_and_price.subproblem import Subproblem
from branch_and_price.subproblem_builder import SubproblemBuilder
from common import TMachineSchedule, is_integer, TAssignment, is_non_zero
from input_data import GeneralAssignmentProblem


class BranchNode:

    next_node_id = itertools.count(start=0)

    def __init__(self,
                 gap_instance: GeneralAssignmentProblem,
                 branching_rules: List[BranchingRule],
                 machine_schedules: List[TMachineSchedule]):

        self.id = next(self.next_node_id)

        self.branching_rules = branching_rules
        self.gap_instance = gap_instance

        self._rmp = grb.Model(f'GAP_RMP_{self.id}')
        self._rmp.Params.LogToConsole = 0

        self.next_machine_schedule_index = itertools.count(start=0)
        self.machine_schedule_index = dict()
        self.machine_schedule_index_to_variable = dict()
        self.machine_to_assignment_constraint: Dict = dict()
        self.task_to_assignment_constraint: Dict = dict()

        self._build_constraints()
        self._add_feasible_initial_columns(machine_schedules)

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
        # improve
        machine_schedule_index_to_variable = bidict(self.machine_schedule_index_to_variable)

        for var in self._rmp.getVars():
            if is_non_zero(var.x):
                assignment_idx = machine_schedule_index_to_variable.inverse.get(var)
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

    def objective_value(self) -> Optional[float]:
        return \
            self._rmp.getAttr(grb.GRB.Attr.ObjVal) \
            if self._rmp.status == grb.GRB.Status.OPTIMAL \
            else float('nan')

    def get_machine_schedules(self):
        return list(self.machine_schedule_index.values())

    def report_solution(self):
        obj_val = self.objective_value()

        logging.info(f"** Solution to RMP on node {self.id}! **")
        logging.info("Objective value: %f", obj_val)

        for var in self._rmp.getVars():
            if is_non_zero(var.x):
                logging.info(f'{var.VarName} \t:{var.X}')

        logging.info('')

    def _solve_using_column_generation(self):
        logging.info("[CG] Solving GAP using column generation")

        self._rmp.setAttr(grb.GRB.Attr.ModelSense, grb.GRB.MAXIMIZE)

        optimal = False
        itr_cnt = itertools.count(start=1)
        itr_with_no_progress_cnt = 0
        subproblem_builder = SubproblemBuilder(gap_instance=self.gap_instance)
        previous_itr_objective_value = math.nan

        while not optimal:
            col_gen_itr = next(itr_cnt)
            logging.debug("[CG] Column generation iteration ... {}".format(col_gen_itr))
            if col_gen_itr % 200 == 0:
                logging.info("[CG] Column generation iteration %d on node %d", col_gen_itr, self.id)
                logging.info("[CG]  * Objective value: {:.2f}".format(self._rmp.getObjective().getValue()))

            self._rmp.update()
            self._rmp.write(self._rmp.ModelName + '.lp')
            self._rmp.optimize()

            if math.isclose(previous_itr_objective_value, self._rmp.getObjective().getValue()):
                itr_with_no_progress_cnt += 1
                if itr_with_no_progress_cnt > 50:
                    logging.info("[CG] Stopping due to no progress."
                                 "Iteration %d on node %d."
                                 "The latest objective value: %.1f",
                                 col_gen_itr, self.id, self._rmp.getObjective().getValue())
                    break
            else:
                previous_itr_objective_value = self._rmp.getObjective().getValue()
            # self.report_solution()
            # obtain duals associated with tasks, solution might be infeasible
            # but duals will be returned
            task_duals = [row.Pi for _, row in self.task_to_assignment_constraint.items()]
            machine_duals = [row.Pi for _, row in self.machine_to_assignment_constraint.items()]
            # self.report_solution()
            optimal = True

            best_subproblem: Optional[Subproblem] = None

            # self.do_stuff()

            for machine_id in range(self.gap_instance.num_machines):
                logging.debug("[CG]  * Solving subproblem for machine {}".format(machine_id))

                machine_dual = machine_duals[machine_id]

                subproblem = subproblem_builder.build(machine_id=machine_id,
                                                      machine_dual=machine_dual,
                                                      task_duals=task_duals,
                                                      branching_rules=self.branching_rules)
                # subproblem._model.write(f'subproblem_{self.id}_{itr_cnt}_{machine_id}.lp')
                subproblem.solve()
                subproblem_objective_value = subproblem.objective_value()
                if subproblem_objective_value is None or subproblem_objective_value <= 0:
                    continue

                optimal = False

                for machine_schedule in subproblem.all_solutions():
                    self._add_column_to_rmp(machine_schedule)

            #     if best_subproblem is None or best_subproblem.objective_value() > subproblem_objective_value:
            #         best_subproblem = subproblem
            #
            # if best_subproblem is not None:
            #
            #     for machine_schedule in best_subproblem.all_solutions():
            #         self._add_column_to_rmp(machine_schedule)

    def do_stuff(self):
        n_tasks = self.gap_instance.num_tasks
        n_rows = self.gap_instance.num_machines + self.gap_instance.num_tasks
        cols = self.get_machine_schedules()
        n_cols = len(cols)
        A = np.zeros(shape=(n_rows, n_cols))

        for machine, tasks in cols:
            col_idx = machine
            row_idx = n_tasks + machine
            A[row_idx][col_idx] = 1.0
            for task in tasks:
                row_idx = task
                A[row_idx][col_idx] = 1.0

        b = np.ones(shape=n_rows)
        bounds = [(None, 1.0) for _ in range(n_cols)]
        c = np.array([-self.gap_instance.machine_schedule_profit(col) for col in cols])

        res = scipy.optimize.linprog(c=c, A_eq=A, b_eq=b, bounds=bounds)
        print(res)

    def _build_constraints(self):
        self._build_task_binding_constraints()
        self._build_machine_binding_constraints()

    def _build_task_binding_constraints(self):

        for task_id in range(self.gap_instance.num_tasks):
            lhs = grb.quicksum([])
            rhs = 1
            name = f'task_assignment_{task_id}'
            c = self._rmp.addConstr(lhs == rhs, name=name)
            self.task_to_assignment_constraint[task_id] = c

    def _build_machine_binding_constraints(self):
        for machine_id in range(self.gap_instance.num_machines):
            lhs = grb.quicksum([])
            rhs = 1
            name = f'convexity_machine_{machine_id}'
            c = self._rmp.addConstr(lhs == rhs, name=name)
            self.machine_to_assignment_constraint[machine_id] = c

    def _add_feasible_initial_columns(self, machine_schedules: List[TMachineSchedule]):
        """
        Filters out columns violating branching rules.
        Then, adds initial columns which are base columns of all parent nodes.
        """
        machine_schedules = self._filter_machine_schedule_based_on_branching_rule(
            self.branching_rules,
            machine_schedules
        )

        for machine_schedule in machine_schedules:
            self._add_column_to_rmp(machine_schedule)

    def _add_column_to_rmp(self, machine_schedule: TMachineSchedule):
        """
        Adds a column that represents assigning `tasks` to `machine_id` to RMP.
        It does so by performing two steps:
        (1) creating variable and adding to Gurobi problem/model.
        (2) storing information about what kind of assignment a column represents.

        :param machine_schedule: machine schedule
        """
        machine_schedule_index = next(self.next_machine_schedule_index)
        self.machine_schedule_index[machine_schedule_index] = copy.deepcopy(machine_schedule)

        machine_id = machine_schedule[0]
        tasks = machine_schedule[1]

        profit = self.gap_instance.machine_schedule_profit(machine_schedule)
        name = f"machine_{machine_id}_tasks_{'_'.join(str(task_id) for task_id in tasks)}"

        c = grb.Column()
        coeff = 1.0
        for task in tasks:
            constr = self.task_to_assignment_constraint[task]
            c.addTerms(coeff, constr)

        machine_constr = self.machine_to_assignment_constraint[machine_id]
        c.addTerms(coeff, machine_constr)

        var = self._rmp.addVar(
            lb=0.0,
            # ub=1.0,
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
            return machine_schedules

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
