"""Process-level ownership for job storage and individual executions."""

import fcntl
from contextlib import contextmanager


class AlreadyOwned(RuntimeError):
    pass


@contextmanager
def exclusive_file(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise AlreadyOwned("Job storage is already owned by another worker") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
