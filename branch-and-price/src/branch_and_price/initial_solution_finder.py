import logging
from typing import Dict, Set, Tuple

import numpy as np
import networkx as nx

from common import TCompleteSchedule
from input_data import GeneralAssignmentProblem


class InitialSolutionFinder:
    """
    Object responsible for finding initial feasible solution to GAP.
    The solution is considered to be feasible if all tasks are assigned
    and each task is assigned to exactly one machine.
    The capacity restriction for each machine is fulfilled.
    """

    def __init__(self, gap_instance: GeneralAssignmentProblem):
        self.gap_instance = gap_instance

        # current assignment - machine to
        self.solution: Dict[int, Set[int]] = dict((machine, set()) for machine in range(self.gap_instance.num_machines))

        # array specifying whether task is used or not
        self.used_tasks = np.full(self.gap_instance.num_tasks, False)

        self.remaining_capacity = self.gap_instance.capacity.copy()

    def find(self) -> TCompleteSchedule:
        self._initial_assignment()
        self._find_assignment_for_remaining_tasks()
        self._verify_solution_feasibility()
        self._report()

        return [
            (machine_id, list(tasks))
            for machine_id, tasks in self.solution.items()
        ]

    def _initial_assignment(self):
        """
        Constructs bipartite graph with two set of nodes:
        (1) tasks (indexed by i)
        (2) machines (indexed by j)
        Weight of arc (i, j) is equal to -weight[i][j].
        Function finds maximum matching.
        :return:
        """
        while True:
            bipartite_graph = nx.Graph()
            task_node_group_ind = 0
            machine_group_node_ind = 1

            unassigned_tasks = [
                task_id
                for task_id in range(self.gap_instance.num_tasks)
                if not self.used_tasks[task_id]
            ]
            if not unassigned_tasks:
                break

            # Add tasks nodes
            bipartite_graph.add_nodes_from(
                [f't-{task_id}' for task_id in unassigned_tasks],
                bipartite=task_node_group_ind
            )

            # Add machine nodes
            bipartite_graph.add_nodes_from(
                [f'm-{machine_id}' for machine_id in range(self.gap_instance.num_machines)],
                bipartite=machine_group_node_ind
            )

            bipartite_graph.add_weighted_edges_from(
                (f't-{task_id}', f'm-{machine_id}', self.gap_instance.weight(task_id=task_id, machine_id=machine_id))
                for task_id in unassigned_tasks
                for machine_id in range(self.gap_instance.num_machines)
                if self.gap_instance.weight(task_id=task_id, machine_id=machine_id) <= self.remaining_capacity[machine_id]
            )

            assert nx.is_bipartite(bipartite_graph)
            minimum_weight_matching = nx.algorithms.min_weight_matching(bipartite_graph)
            if not minimum_weight_matching:
                break

            def _unwrap(assignment) -> Tuple[int, int]:
                first, second = assignment
                if first[0] == 't':
                    return int(first[2:]), int(second[2:])
                else:
                    return int(second[2:]), int(first[2:])

            for assignment in minimum_weight_matching:
                task_id, machine_id = _unwrap(assignment)
                self._assign(machine_id, task_id)

    def _select_unassigned_task(self) -> int:
        """
        Returns first unassigned task.
        """
        return np.nonzero(self.used_tasks == False)[0][0]

    def _select_machine(self, task) -> int:
        """
        Identifies machine such that task is contributing the least to weight
        :param task:
        :return: int
        """

        arr = self.gap_instance.weights[:, task]
        return np.argmin(arr)

    def _free_capacity(self, machine_id: int, weight: float):
        """
        For a machine it de-assigns already assigned tasks to
        make room for task(s) with given weight. Tasks assigned
        to machine are randomly shuffled and they are de-assigned
        one by one from the end until there is enough capacity.
        :param machine_id: machine from which tasks will be unassigned
        :param weight: the amount of needed space to free
        """

        assigned_tasks = list(self.solution[machine_id])
        np.random.shuffle(assigned_tasks)

        while weight <= self.remaining_capacity[machine_id] or not assigned_tasks:
            task_to_remove = assigned_tasks.pop()
            self._de_assign(machine_id, task_to_remove)

    def _de_assign(self, machine, task):
        weight = self.gap_instance.weight(task_id=task, machine_id=machine)
        self.solution[machine].remove(task)
        self.used_tasks[task] = False
        self.remaining_capacity[machine] += weight

    def _assign(self, machine, task):
        weight = self.gap_instance.weight(task_id=task, machine_id=machine)
        self.used_tasks[task] = True
        self.solution[machine].add(task)
        self.remaining_capacity[machine] -= weight

    def _find_assignment_for_remaining_tasks(self):

        # try to fit remaining unassigned tasks as follows
        # 1. Get unassigned task
        # 2. Identify machine where task can be assigned
        # 3. Make room for and assign task.
        # 4. Continue until no remaining tasks
        # The question is whether a procedure guarantees to finish.
        # Potential cycles?
        solution_found = (np.sum(self.used_tasks) == self.gap_instance.num_tasks)
        while not solution_found:
            task = self._select_unassigned_task()
            machine = self._select_machine(task)

            weight = self.gap_instance.weight(task_id=task, machine_id=machine)
            if weight < self.gap_instance.machine_capacity(machine_id=machine):
                # does not make sense to free capacity is task cannot fit into machine
                continue

            # now de-assign some tasks to accommodate task
            self._free_capacity(machine, weight)
            self._assign(machine, task)

            solution_found = (np.sum(self.used_tasks) == self.gap_instance.num_tasks)

    def _verify_solution_feasibility(self):
        feasible = True
        used_tasks = np.full(self.gap_instance.num_tasks, False)

        for machine, tasks in self.solution.items():
            allocated_weight = 0
            for task in tasks:
                allocated_weight += self.gap_instance.weight(task_id=task, machine_id=machine)

                if used_tasks[task] is True:
                    logging.error(f"Task {task} assigned to two machines.")
                    feasible = False
                    break
                used_tasks[task] = True

            if not feasible:
                break

            feasible = allocated_weight <= self.gap_instance.capacity[machine]
            if not feasible:
                logging.error("Solution not feasible for machine {%d} - allocated weight {%f} > capacity {%f}",
                              machine, allocated_weight, self.gap_instance.machine_capacity(machine))
                break

        if not feasible:
            return

        feasible = np.sum(used_tasks) == self.gap_instance.num_tasks
        if not feasible:
            logging.error("Not all tasks have been assigned.")

    def _report(self):

        tot_profit = 0
        logging.info(" * Initial feasible solution")
        for machine, tasks in self.solution.items():
            tot_profit += np.sum(self.gap_instance.profits[machine, task] for task in tasks)
            s = "\t{}: {}".format(machine, " ".join([str(task) for task in tasks]))
            logging.info(s)
        logging.info(" ** Associated profit: {}".format(tot_profit))
