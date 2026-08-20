"""Pytest configuration shared by the Keras tests."""

import os

# Keras resolves its backend once, while it is first imported, and never switches afterwards, so the
# choice has to be made before any keras import reaches the interpreter -- hence at module level of
# this conftest rather than in a fixture. Without it these tests would run against whatever
# ~/.keras/keras.json names on the machine, which decides which framework every op below executes
# on. `setdefault` leaves an explicitly requested backend alone, so the same tests can be run
# against jax or torch by exporting KERAS_BACKEND.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
