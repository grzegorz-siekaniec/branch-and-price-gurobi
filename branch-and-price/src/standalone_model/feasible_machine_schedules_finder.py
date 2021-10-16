from copy import copy
from typing import List

import numpy as np

from common import TMachineSchedule, TTask
from input_data import GeneralAssignmentProblem


class FeasibleMachineSchedulesFinder:
    """
    Object responsible for finding all feasible assignments of
    tasks to machine.
    """

    def __init__(self, gap_input: GeneralAssignmentProblem):
        self.gap_input = gap_input

    def find(self) -> List[TMachineSchedule]:

        all_feasible_assignments = []
        for machine_id in range(self.gap_input.num_machines):
            all_feasible_assignments.extend(self._find_machine(machine_id))

        return all_feasible_assignments

    def _find_machine(self, machine_id) -> List[TMachineSchedule]:
        """
        Find all feasible assignment for a machine
        :param machine_id: machine id
        :return: feasible assignments for machine
        """
        assignments = self._MachineSolutionFinder(
            machine_capacity=self.gap_input.machine_capacity(machine_id),
            num_tasks=self.gap_input.num_tasks,
            task_weights=self.gap_input.weights[machine_id]).find()

        return [
            (machine_id, assignment)
            for assignment in assignments
        ]

    class _MachineSolutionFinder:

        def __init__(self, machine_capacity: float, num_tasks: int, task_weights: np.ndarray):
            self.num_tasks = num_tasks
            self.task_weights = task_weights
            self.machine_capacity = machine_capacity

            self.feasible_assignments: List[List[TTask]] = []

        def find(self) -> List[List[int]]:

            assignment = []
            for task_id in range(self.num_tasks):
                assignment.append(task_id)
                self._find(next_task_id=task_id+1, assignment=assignment)
                assignment.pop()

            return self.feasible_assignments

        def _find(self, next_task_id, assignment):

            if not self.is_feasible(assignment):
                return

            self.feasible_assignments.append(copy(assignment))

            if next_task_id >= self.num_tasks:
                return

            for task_id in range(next_task_id, self.num_tasks):
                assignment.append(task_id)
                self._find(next_task_id=task_id + 1, assignment=assignment)
                assignment.pop()

        def is_feasible(self, assignment):
            assignment_weights = [
                self.task_weights[task_id]
                for task_id in assignment
            ]
            assignment_weight = np.sum(assignment_weights)
            return assignment_weight <= self.machine_capacity

