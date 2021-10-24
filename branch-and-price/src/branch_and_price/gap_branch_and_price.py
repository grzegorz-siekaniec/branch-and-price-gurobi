import copy
import logging
from typing import Optional, Tuple
import networkx as nx
from matplotlib import pyplot

from branch_and_price.branch_node import BranchNode
from branch_and_price.branching_rule import BranchingRule
from branch_and_price.initial_solution_finder import InitialSolutionFinder
from common.queue import Queue
from input_data import GeneralAssignmentProblem


class GAPBranchAndPrice:

    def __init__(self, gap_instance: GeneralAssignmentProblem):
        self.gap_instance = gap_instance
        self.tree = nx.DiGraph()

    def solve(self):
        queue: Queue[BranchNode] = Queue([
            self._create_root_node()
        ])

        best_solution_node = None
        mip_lb = None

        while not queue.is_empty():

            current_node = queue.pop()

            logging.info("[BAP] Processing node {}.".format(current_node.id))

            current_node.solve()

            if not current_node.is_feasible():
                logging.info("[BAP] Solution at node {} is infeasible.".format(current_node.id))
                continue

            if current_node.has_integer_solution():
                logging.info("[B&P] Solution at node {} has integer solution.".format(current_node.id))
                obj = current_node.objective_value()
                current_node.report_solution()
                if mip_lb is None or obj > mip_lb:
                    best_solution_node = current_node
                    mip_lb = obj
            else:
                obj = current_node.objective_value()
                logging.info("[B&P] Solution at node %d has non integer solution. Obj %.1f", current_node.id, obj)
                if nodes := self._branch(current_node, mip_lb):
                    include_nd, exclude_nd = nodes
                    queue.push(include_nd)
                    queue.push(exclude_nd)

                    self.tree.add_edge(current_node.id, include_nd.id)
                    self.tree.add_edge(current_node.id, exclude_nd.id)

        from networkx.drawing.nx_agraph import graphviz_layout

        pos = graphviz_layout(self.tree, prog='dot')
        nx.draw(self.tree, pos, with_labels=True, arrows=True)
        pyplot.show()
        best_solution_node.report_solution()

    def _create_root_node(self):
        self.tree.add_node(0)
        initial_solution = InitialSolutionFinder(self.gap_instance).find()
        branching_rules = []
        return BranchNode(
            gap_instance=self.gap_instance,
            branching_rules=branching_rules,
            machine_schedules=initial_solution
        )

    @classmethod
    def _branch(cls, node: BranchNode, mip_lb: float) -> Optional[Tuple[BranchNode, BranchNode]]:
        """
        Branches based on results from `node`. It obtains non-integers
        variable from solution to `node` and associated machine and task.
        Then it creates two new nodes:
        (1) Forbidding assigning task to machine.
        (2) Forcing assigning task to machine.
        Two new nodes created based on those branching strategies are added to queue.
        """

        # in case node's LP value is lower than
        # so far found MIP LB, then whole tree rooted at node
        # can be discarded
        if mip_lb is not None and node.objective_value() <= mip_lb:
            return None

        # based on current solution obtain id of task and machine
        machine, task = node.machine_task_to_branch_on()
        logging.info("[BAP] Current node {}. Branching on machine {} and task {}".format(node.id, machine, task))

        # create two branching rules
        exclude_branching = BranchingRule(task, machine, assigned=False)
        include_branching = BranchingRule(task, machine, assigned=True)

        # current branching rules
        br_rls = node.branching_rules

        exclude_nd = BranchNode(
            node.gap_instance,
            copy.deepcopy(br_rls) + [exclude_branching],
            copy.deepcopy(node.get_machine_schedules()),
        )

        include_nd = BranchNode(
            node.gap_instance,
            copy.deepcopy(br_rls) + [include_branching],
            copy.deepcopy(node.get_machine_schedules()),
        )

        logging.info("  Exclude node {}".format(exclude_nd.id))
        logging.info("  Include node {}".format(include_nd.id))

        return exclude_nd, include_nd
