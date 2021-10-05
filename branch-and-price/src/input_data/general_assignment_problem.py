from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GeneralAssignmentProblem:

    num_tasks: int
    num_machines: int
    weights: np.ndarray
    profits: np.ndarray
    capacity: np.ndarray

    def profit(self, task_id: int, machine_id: int) -> float:
        return self.profits[machine_id][task_id]

    def weight(self, task_id: int, machine_id: int) -> float:
        return self.weights[machine_id][task_id]

    def machine_capacity(self, machine_id: int) -> float:
        return self.capacity[machine_id]


def small_example() -> GeneralAssignmentProblem:
    num_machines = 2
    num_tasks = 7
    profits = np.array([
        [6, 9, 4, 2, 10, 3, 6],
        [4, 8, 9, 1, 7, 5, 4]
    ])

    weights = np.array([
        [4, 1, 2, 1, 4, 3, 8],
        [9, 9, 8, 1, 3, 8, 7]
    ])

    capacity = np.array([11, 22])
    return GeneralAssignmentProblem(
        num_tasks=num_tasks,
        num_machines=num_machines,
        weights=weights,
        profits=profits,
        capacity=capacity
    )


def medium_example() -> GeneralAssignmentProblem:
    num_machines = 8
    num_tasks = 24
    weights = np.array([
        [8, 18, 22, 5, 11, 11, 22, 11, 17, 22, 11, 20, 13, 13, 7, 22, 15, 22, 24, 8, 8, 24, 18, 8],
        [24, 14, 11, 15, 24, 8, 10, 15, 19, 25, 6, 13, 10, 25, 19, 24, 13, 12, 5, 18, 10, 24, 8, 5],
        [22, 22, 21, 22, 13, 16, 21, 5, 25, 13, 12, 9, 24, 6, 22, 24, 11, 21, 11, 14, 12, 10, 20, 6],
        [13, 8, 19, 12, 19, 18, 10, 21, 5, 9, 11, 9, 22, 8, 12, 13, 9, 25, 19, 24, 22, 6, 19, 14],
        [25, 16, 13, 5, 11, 8, 7, 8, 25, 20, 24, 20, 11, 6, 10, 10, 6, 22, 10, 10, 13, 21, 5, 19],
        [19, 19, 5, 11, 22, 24, 18, 11, 6, 13, 24, 24, 22, 6, 22, 5, 14, 6, 16, 11, 6, 8, 18, 10],
        [24, 10, 9, 10, 6, 15, 7, 13, 20, 8, 7, 9, 24, 9, 21, 9, 11, 19, 10, 5, 23, 20, 5, 21],
        [6, 9, 9, 5, 12, 10, 16, 15, 19, 18, 20, 18, 16, 21, 11, 12, 22, 16, 21, 25, 7, 14, 16, 10]
    ])
    profits = np.array([
        [25, 23, 20, 16, 19, 22, 20, 16, 15, 22, 15, 21, 20, 23, 20, 22, 19, 25, 25, 24, 21, 17, 23, 17],
        [16, 19, 22, 22, 19, 23, 17, 24, 15, 24, 18, 19, 20, 24, 25, 25, 19, 24, 18, 21, 16, 25, 15, 20],
        [20, 18, 23, 23, 23, 17, 19, 16, 24, 24, 17, 23, 19, 22, 23, 25, 23, 18, 19, 24, 20, 17, 23, 23],
        [16, 16, 15, 23, 15, 15, 25, 22, 17, 20, 19, 16, 17, 17, 20, 17, 17, 18, 16, 18, 15, 25, 22, 17],
        [17, 23, 21, 20, 24, 22, 25, 17, 22, 20, 16, 22, 21, 23, 24, 15, 22, 25, 18, 19, 19, 17, 22, 23],
        [24, 21, 23, 17, 21, 19, 19, 17, 18, 24, 15, 15, 17, 18, 15, 24, 19, 21, 23, 24, 17, 20, 16, 21],
        [18, 21, 22, 23, 22, 15, 18, 15, 21, 22, 15, 23, 21, 25, 25, 23, 20, 16, 25, 17, 15, 15, 18, 16],
        [19, 24, 18, 17, 21, 18, 24, 25, 18, 23, 21, 15, 24, 23, 18, 18, 23, 23, 16, 20, 20, 19, 25, 21]
    ])
    capacity = np.array([36, 35, 38, 34, 32, 34, 31, 34])
    return GeneralAssignmentProblem(
        num_tasks=num_tasks,
        num_machines=num_machines,
        weights=weights,
        profits=profits,
        capacity=capacity
    )
