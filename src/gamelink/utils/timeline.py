import bisect
from collections.abc import Iterable


class Timeline[TNode]:
    """A linear sequence of nodes with a movable cursor and checkpoint support. \
    Enforces a linear history: new nodes can only be added at the tip.
    """

    def __init__(self, root: TNode) -> None:
        self._timeline: list[TNode] = [root]
        self._checkpoints: list[int] = []
        self._cursor: int = 0

    @property
    def current(self) -> TNode:
        """Return the node at the current cursor position."""
        return self._timeline[self._cursor]

    @property
    def future(self) -> Iterable[TNode]:
        """Return all nodes after the current cursor position."""
        return self._timeline[self._cursor :]

    def seek(self, index: int) -> None:
        """Move the cursor to a specific index."""
        if not (0 <= index < len(self._timeline)):
            msg = "Timeline index out of range"
            raise IndexError(msg)
        self._cursor = index

    def step_forward(self) -> None:
        """Advance the cursor by one."""
        self.seek(self._cursor + 1)

    def prune_future(self) -> None:
        """Delete all nodes coming after the current cursor."""
        self.truncate_at(self._cursor)

    def checkpoint(self) -> None:
        """Mark the current end of the timeline as a checkpoint."""
        self._checkpoints.append(len(self._timeline) - 1)

    def pop_checkpoint(self) -> int:
        """Remove and returns the index of the most recent checkpoint."""
        return self._checkpoints.pop()

    def append(self, node: TNode) -> None:
        """Add a node to the end of the timeline. \
        Raise an error if the cursor is not at the end of the timeline.
        """
        if self._cursor != len(self._timeline) - 1:
            msg = "Cannot append when cursor is not at the timeline tip."
            raise RuntimeError(msg)

        self._timeline.append(node)
        # Automatically advance cursor to the new tip
        self._cursor += 1

    def truncate_at(self, index: int) -> None:
        """Remove all nodes and checkpoints strictly after the given index. Move \
        cursor to the index if it was previously ahead of it.
        """
        # Keep elements up to `index`, delete `index + 1` onwards
        del self._timeline[index + 1 :]

        # Remove checkpoints that are now beyond the end of the timeline
        cut_point = bisect.bisect_right(self._checkpoints, index)
        del self._checkpoints[cut_point:]

        # Ensure cursor remains valid
        self._cursor = min(len(self._timeline) - 1, self._cursor)
