"""Executable, translation-invariant generalized Ising expressions.

Only NumPy is required. Coordinates are (row, column); each translated term is
counted once for every lattice origin, exactly as in the stated Hamiltonian.
"""
from dataclasses import dataclass
from itertools import combinations, product
import math

import numpy as np

SIDE = 16


def canonical_structure(points, side=SIDE):
    """Reduce s_i**2=1 and translations on the torus, preserving orientation.

    Empty output is a constant energy term, invisible to flip measurements.
    Reflection/rotation are deliberately NOT equivalences.
    """
    odd = set()
    for point in points:
        if len(point) != 2 or any(not isinstance(v, (int, np.integer)) for v in point):
            raise ValueError('Offsets must be integer coordinate pairs.')
        point = tuple(int(v) % side for v in point)
        if point in odd:
            odd.remove(point)
        else:
            odd.add(point)
    if not odd:
        return ()
    return min(tuple(sorted(((x-a) % side, (y-b) % side) for x, y in odd))
               for a, b in odd)


def check_query(configuration, position, side=SIDE):
    state = np.asarray(configuration)
    if state.shape != (side, side) or not np.all((state == 1) | (state == -1)):
        raise ValueError(f'Configuration must be a {side}x{side} array of -1/+1.')
    if len(position) != 2 or any(not isinstance(v, (int, np.integer)) or not 0 <= v < side
                                 for v in position):
        raise ValueError('Position must contain two in-range integers.')
    return state.astype(np.int8, copy=True), tuple(int(v) for v in position)


def base_flip(configuration, position):
    state, (x, y) = check_query(configuration, position)
    neighbours = (int(state[(x-1) % SIDE, y]) + int(state[(x+1) % SIDE, y])
                  + int(state[x, (y-1) % SIDE]) + int(state[x, (y+1) % SIDE]))
    return 2.0 * int(state[x, y]) * neighbours


@dataclass(frozen=True)
class Term:
    points: tuple
    coefficient: float


@dataclass(frozen=True)
class Expression:
    """Additional terms; H0 is always included implicitly."""
    terms: tuple = ()

    def __post_init__(self):
        merged = {}
        for term in self.terms:
            points = canonical_structure(term.points)
            coefficient = float(term.coefficient)
            if not math.isfinite(coefficient):
                raise ValueError('Coefficients must be finite.')
            if points:
                merged[points] = merged.get(points, 0.0) + coefficient
        object.__setattr__(self, 'terms', tuple(Term(p, c) for p, c in sorted(merged.items())
                                               if abs(c) > 1e-10))

    @property
    def support(self):
        return tuple(t.points for t in self.terms)

    def to_dict(self):
        return {'base': 'nearest_neighbour_J=1', 'terms': [
            {'offsets': [list(p) for p in t.points], 'coefficient': t.coefficient}
            for t in self.terms]}

    def delta(self, configuration, position):
        """Direct local implementation, independent of compiled search features."""
        state, (x, y) = check_query(configuration, position)
        result = base_flip(state, (x, y))
        for term in self.terms:
            for ax, ay in term.points:
                value = 1
                for dx, dy in term.points:
                    value *= int(state[(x-ax+dx) % SIDE, (y-ay+dy) % SIDE])
                result += 2.0 * term.coefficient * value
        return result

    def energy(self, configuration):
        """Full lattice sum for independent energy-difference verification."""
        state, _ = check_query(configuration, (0, 0))
        result = -float(np.sum(state * (np.roll(state, -1, 0) + np.roll(state, -1, 1))))
        for term in self.terms:
            values = np.ones_like(state)
            for dx, dy in term.points:
                values = values * np.roll(state, (-dx, -dy), (0, 1))
            result -= term.coefficient * float(values.sum())
        return result


class Grammar:
    """Enumerate syntax, NOT the hidden generator's list of mechanisms.

    Any nonempty subset of a 3x3 window, up to order four, is allowed. Translation
    canonicalization yields 158 structures (1/12/48/97 by order).
    """
    def __init__(self, window=3, max_order=4):
        if not 1 <= window <= SIDE // 2 or not 1 <= max_order <= 4:
            raise ValueError('Use window <= L/2 and maximum order 1..4.')
        points = list(product(range(window), repeat=2))
        structures = {canonical_structure(c) for k in range(1, max_order+1)
                      for c in combinations(points, k)}
        self.structures = tuple(sorted(structures, key=lambda a: (len(a), a)))
        self.index = {p: j for j, p in enumerate(self.structures)}
        self.window = window
        self.max_order = max_order
        count = len(self.structures)
        self.offsets = np.zeros((count, max_order, max_order, 2), dtype=int)
        self.factors = np.zeros((count, max_order, max_order), dtype=bool)
        self.anchors = np.zeros((count, max_order), dtype=bool)
        for j, structure in enumerate(self.structures):
            for a, anchor in enumerate(structure):
                self.anchors[j, a] = True
                for d, point in enumerate(structure):
                    self.offsets[j, a, d] = np.subtract(point, anchor)
                    self.factors[j, a, d] = True

    def features(self, configurations, positions):
        states = np.asarray(configurations)
        positions = np.asarray(positions)
        if states.ndim != 3 or states.shape[1:] != (SIDE, SIDE):
            raise ValueError('Expected a batch of 16x16 states.')
        if positions.shape != (len(states), 2) or not np.issubdtype(positions.dtype, np.integer):
            raise ValueError('Expected an integer position for each state.')
        if np.any((states != 1) & (states != -1)) or np.any((positions < 0) | (positions >= SIDE)):
            raise ValueError('Invalid spin or position.')
        indices = (positions[:, None, None, None, :] + self.offsets[None]) % SIDE
        values = states[np.arange(len(states))[:, None, None, None], indices[..., 0], indices[..., 1]]
        values = np.where(self.factors[None], values, 1)
        return 2.0 * (values.prod(axis=-1) * self.anchors[None]).sum(axis=-1)

    def coefficients(self, expression):
        result = np.zeros(len(self.structures))
        for term in expression.terms:
            if term.points not in self.index:
                raise ValueError('Expression is outside the public grammar.')
            result[self.index[term.points]] = term.coefficient
        return result

    def expression(self, support, coefficients):
        return Expression(tuple(Term(self.structures[int(j)], float(c))
                                for j, c in zip(support, coefficients)))


def random_queries(rng, count):
    states = rng.choice(np.array([-1, 1], dtype=np.int8), size=(count, SIDE, SIDE))
    positions = rng.integers(SIDE, size=(count, 2))
    return [(state, tuple(int(v) for v in position)) for state, position in zip(states, positions)]


def encode_query(query):
    state, position = query
    return {'rows': [''.join('+' if v == 1 else '-' for v in row) for row in state],
            'position': list(position)}
