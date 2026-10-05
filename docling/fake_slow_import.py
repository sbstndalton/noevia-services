"""A conversion target whose module takes too long to import: a worker that never becomes ready."""
import time

time.sleep(60)


def extract(path, name):  # pragma: no cover - never reached
    return {}
