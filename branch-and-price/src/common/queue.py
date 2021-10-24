from typing import Generic, List, TypeVar, Collection

T = TypeVar("T")


class Queue(Generic[T]):

    def __init__(self, lst: Collection[T] = None):
        self._queue: List[T] = list(lst) if lst else list()

    def push(self, el: T):
        """

        :param el:
        """
        self._queue.append(el)

    def pop(self) -> T:
        return self._queue.pop(0)

    def is_empty(self) -> bool:
        return not self._queue
