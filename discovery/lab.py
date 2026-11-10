"""Prediction-before-observation protocol and metered laboratory.

This is an API boundary for cooperative algorithms, not an adversarial Python
sandbox. Future LLM agents must run outside the process holding the oracle.
"""
from copy import deepcopy
import math

import numpy as np

from .physics import check_query, encode_query


class BudgetExceeded(ValueError):
    pass


class Laboratory:
    def __init__(self, oracle, *, noise_sigma=0.0, noise_seed=0, budget=32, initial_count=8):
        if not math.isfinite(noise_sigma) or noise_sigma < 0:
            raise ValueError('Noise sigma must be finite and nonnegative.')
        if not isinstance(budget, int) or not isinstance(initial_count, int) or not 0 < initial_count <= budget:
            raise ValueError('Invalid budget or initial count.')
        self.__oracle = oracle
        self.__rng = np.random.default_rng(noise_seed)
        self.noise_sigma = float(noise_sigma)
        self.budget = budget
        self.initial_count = initial_count
        self.used = 0
        self.__pending = None
        self.__next_ticket = 0
        self.__events = []

    @property
    def events(self):
        return deepcopy(self.__events)

    def __measure(self, query):
        state, position = query
        clean = self.__oracle(state, position)
        observed = float(clean + self.__rng.normal(0, self.noise_sigma))
        self.used += 1
        return observed

    def initial_observations(self, queries):
        if self.used or len(queries) != self.initial_count:
            raise ValueError('Exactly one shared initial batch is allowed.')
        checked = [check_query(s, p) for s, p in queries]
        values = [self.__measure(q) for q in checked]
        self.__events.append({'kind': 'initial', 'queries': [encode_query(q) for q in checked],
                              'observed': values, 'used': self.used})
        # Do not expose the list retained in the audit event. Callers may append
        # later observations to their copy without rewriting the initial batch.
        return list(values)

    def commit(self, queries, models, predictions, reason, update_rule):
        """Lock expressions, exact numerical predictions, and queries before reveal."""
        if self.used < self.initial_count or self.__pending is not None:
            raise ValueError('Initialize first; reveal the pending experiment before committing another.')
        if not queries or self.used + len(queries) > self.budget:
            raise BudgetExceeded('Each measured flip costs one query, including repeats.')
        checked = [check_query(s, p) for s, p in queries]
        if not models or not reason or not update_rule:
            raise ValueError('Submit a model, experiment reason, and update condition.')
        declared = np.asarray(predictions, dtype=float)
        expected = np.array([[model.delta(s, p) for s, p in checked] for model in models])
        if declared.shape != expected.shape or not np.all(np.isfinite(declared)) or not np.allclose(
                declared, expected, rtol=1e-10, atol=1e-9):
            raise ValueError('Declared predictions do not match executable expressions.')
        ticket = self.__next_ticket
        self.__next_ticket += 1
        # All mutable caller data are copied BEFORE the first oracle call.
        record = {'kind': 'experiment', 'ticket': ticket,
                  'queries': [encode_query(q) for q in checked],
                  'hypotheses': [m.to_dict() for m in models], 'predictions': declared.tolist(),
                  'reason': str(reason), 'update_rule': str(update_rule),
                  'committed_at_used': self.used}
        self.__pending = (ticket, checked, deepcopy(record))
        return ticket

    def reveal(self, ticket):
        if self.__pending is None or ticket != self.__pending[0]:
            raise ValueError('A matching committed experiment is required; tickets are single-use.')
        _, queries, record = self.__pending
        values = [self.__measure(q) for q in queries]
        record.update(observed=values, used=self.used)
        self.__events.append(record)
        self.__pending = None
        return values
