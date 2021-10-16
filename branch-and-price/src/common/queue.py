from typing import Generic, List, TypeVar

T = TypeVar("T")


class Queue(Generic[T]):

    def __init__(self):
        self._queue: List[T] = []

    def push(self, el: T):
        self._queue.append(el)

    def pop(self) -> T:
        return self._queue.pop()

    def is_empty(self) -> bool:
        return len(self._queue) > 0
