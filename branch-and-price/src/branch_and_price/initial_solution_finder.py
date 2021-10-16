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

    def find(self) -> TCompleteSchedule:
        return TCompleteSchedule()