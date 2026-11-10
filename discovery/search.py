"""Non-language-model baseline: exhaustive <=2-term fitting plus query selection.

The search dictionary is generated from the public expression grammar. No world
type, coefficients, evaluation queries, or oracle source are used for selection.
"""
from dataclasses import dataclass

import numpy as np

from .physics import base_flip, random_queries


@dataclass
class Candidate:
    support: tuple
    coefficients: np.ndarray
    sse: float
    score: float


class SparseSearch:
    def __init__(self, grammar, noise_sigma=0.0, committee_size=16):
        self.grammar = grammar
        self.sigma = noise_sigma
        self.committee_size = committee_size
        self.left, self.right = np.triu_indices(len(grammar.structures), 1)

    def fit(self, features, residuals):
        """Fit every empty/single/pair support with least squares; rank by penalized SSE.

        The noiseless penalty only resolves floating-point ties in favour of
        simpler expressions. Noise support is experimental, not calibrated here.
        Rank-one pairs use their minimum-norm solution, not an oracle tie break.
        """
        x = np.asarray(features, dtype=float)
        y = np.asarray(residuals, dtype=float)
        # Every least-squares minimum-norm coefficient is zero in this case.
        # Avoid enumerating thousands of identical empty expressions.
        if not np.any(y):
            return [Candidate((), np.array([]), 0.0, 0.0)]
        gram = x.T @ x
        rhs = x.T @ y
        yy = float(y @ y)
        diagonal = gram.diagonal()
        one_coef = np.divide(rhs, diagonal, out=np.zeros_like(rhs), where=diagonal > 0)
        one_sse = np.maximum(0, yy - rhs * one_coef)
        a, b = diagonal[self.left], diagonal[self.right]
        c = gram[self.left, self.right]
        u, v = rhs[self.left], rhs[self.right]
        det = a*b - c*c
        rank_two = det > 1e-10 * np.maximum(a*b, 1)
        p = np.zeros_like(det)
        q = np.zeros_like(det)
        p[rank_two] = (b*u-c*v)[rank_two] / det[rank_two]
        q[rank_two] = (a*v-c*u)[rank_two] / det[rank_two]
        rank_one = ~rank_two & ((a+b) > 0)
        trace_sq = (a+b)[rank_one]**2
        p[rank_one] = (a*u+c*v)[rank_one] / trace_sq
        q[rank_one] = (c*u+b*v)[rank_one] / trace_sq
        pair_sse = np.maximum(0, yy - 2*(u*p+v*q) + a*p*p + 2*c*p*q + b*q*q)
        # Fixed and truth-independent; no threshold tuned on held-out evaluation.
        penalty = max(1e-8, self.sigma**2 * (2*np.log(len(diagonal)) + np.log(len(y))))
        losses = np.concatenate(([yy], one_sse, pair_sse))
        sizes = np.concatenate(([0], (abs(one_coef) > 1e-10).astype(int),
                                (abs(p) > 1e-10).astype(int)+(abs(q) > 1e-10).astype(int)))
        scores = losses + penalty*sizes
        order = np.argsort(scores, kind='stable')
        result = []
        seen = set()
        for index in order:
            if index == 0:
                support, coefficients = (), np.array([])
            elif index <= len(diagonal):
                support, coefficients = (int(index-1),), np.array([one_coef[index-1]])
            else:
                pair = index-len(diagonal)-1
                support = (int(self.left[pair]), int(self.right[pair]))
                coefficients = np.array([p[pair], q[pair]])
            mask = abs(coefficients) > 1e-10
            support = tuple(j for j, keep in zip(support, mask) if keep)
            coefficients = coefficients[mask]
            key = (support, tuple(np.round(coefficients, 9)))
            if key in seen:
                continue
            seen.add(key)
            result.append(Candidate(support, coefficients, float(losses[index]), float(scores[index])))
            if len(result) == self.committee_size:
                break
        return result

    def expression(self, candidate):
        return self.grammar.expression(candidate.support, candidate.coefficients)

    def choose(self, features, committee, observed_features):
        scale = max(self.sigma**2, committee[0].sse / len(observed_features), 1e-8)
        plausible = [c for c in committee if c.score <= committee[0].score + 9*scale]
        predictions = np.array([features[:, c.support] @ c.coefficients for c in plausible])
        utility = predictions.var(axis=0)
        if self.sigma > 0:
            for candidate in plausible:
                if candidate.support:
                    old = observed_features[:, candidate.support]
                    new = features[:, candidate.support]
                    covariance = np.linalg.pinv(old.T @ old)
                    utility += self.sigma**2 * np.einsum('ni,ij,nj->n', new, covariance, new) / len(plausible)
        # The pool is freshly random; index zero is an unbiased tie fallback.
        return int(np.argmax(utility)) if utility.max() > 1e-12 else 0


def run_search(lab, grammar, initial_queries, *, strategy, seed, pool_size=128):
    """Only the laboratory interface and public grammar reach the algorithm."""
    if strategy not in ('random', 'active') or pool_size < 1:
        raise ValueError('Invalid strategy or pool size.')
    search = SparseSearch(grammar, noise_sigma=lab.noise_sigma)
    rng = np.random.default_rng(seed)
    queries = list(initial_queries)
    observations = list(lab.initial_observations(queries))
    features = grammar.features([q[0] for q in queries], [q[1] for q in queries])
    residuals = np.array([y-base_flip(*q) for q, y in zip(queries, observations)])
    snapshots = {}
    while True:
        committee = search.fit(features, residuals)
        best = search.expression(committee[0])
        snapshots[lab.used] = best
        if lab.used == lab.budget:
            break
        pool = random_queries(rng, pool_size)
        pool_features = grammar.features([q[0] for q in pool], [q[1] for q in pool])
        choice = search.choose(pool_features, committee, features) if strategy == 'active' else 0
        query = pool[choice]
        # Alternatives are ranked candidates, not an assertion that they remain plausible.
        models = [search.expression(c) for c in committee[:3]]
        predictions = [[m.delta(*query)] for m in models]
        ticket = lab.commit([query], models, predictions,
                            'Largest prediction disagreement among near-best fits in a random pool.'
                            if strategy == 'active' else 'Uniform random configuration and position.',
                            'Refit all <=2-term supports after observation; rank by penalized residual error.')
        value = lab.reveal(ticket)[0]
        queries.append(query)
        observations.append(value)
        features = np.vstack((features, pool_features[choice]))
        residuals = np.append(residuals, value-base_flip(*query))
    return snapshots, lab.events
