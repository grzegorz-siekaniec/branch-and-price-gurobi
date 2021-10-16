import dataclasses


@dataclasses.dataclass(frozen=True)
class BranchingRule:
    """Represents branching rule for GAP problem.

    There are two types of branching rules:
    (1) A task must be assigned to a machine.
    (2) A task must not be assigned to a machine.
    """
    
    task: int
    machine: int
    # indicator variable specifying whether
    #  * task must be assigned to machine `machine` or whether
    #  * task must *not* be assigned to machine `machine`
    assigned: bool
