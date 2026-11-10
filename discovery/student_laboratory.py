"""Public JSON boundary for a student proposing and testing mechanisms.

Students supply positions and experiments; shared arithmetic fits ONLY their
proposed supports to already revealed observations. No oracle, true mechanism,
teacher committee or candidate list is exposed through public replies.
"""
from copy import deepcopy
import hashlib
import json
import math

import numpy as np

from .physics import Expression, SIDE, Term, base_flip, check_query, encode_query
from .student_spins import OFFSETS, local_query, public_coverage
from .student_v2 import hex_query, packed
from .student_v3 import canonical_proposal
from .teaching import decode_query


def public_query(query):
    return local_query(hex_query(encode_query(query)))


def decode_student_query(value):
    if not isinstance(value, dict):
        raise ValueError('Experiment must be a JSON object.')
    if set(value) == {'rows', 'position'}:
        # Reject Python bool coordinates before the physics integer validator.
        if not isinstance(value['position'], list) or any(type(v) is not int for v in value['position']):
            raise ValueError('Position must be an integer pair.')
        return decode_query(value)
    if set(value) != {'local_spins', 'position'}:
        raise ValueError('Use full rows or a 5x5 local patch plus position.')
    position = value['position']
    if (not isinstance(position, list) or len(position) != 2
            or any(type(v) is not int or not 0 <= v < SIDE for v in position)):
        raise ValueError('Position must be an in-range integer pair.')
    patch = value['local_spins']
    if (not isinstance(patch, list) or len(patch) != 5
            or any(not isinstance(row, list) or len(row) != 5 for row in patch)
            or any(type(v) is not int or v not in (-1, 1) for row in patch for v in row)):
        raise ValueError('Patch must contain exactly 25 integer -1/+1 values.')
    state = np.ones((SIDE, SIDE), dtype=np.int8)
    for i, dx in enumerate(OFFSETS):
        for j, dy in enumerate(OFFSETS):
            state[(position[0]+dx) % SIDE, (position[1]+dy) % SIDE] = patch[i][j]
    return check_query(state, position)


def visible_queries(evidence):
    if set(evidence) != {'measurement_noise_sigma', 'used', 'measurements'}:
        raise ValueError('Only public observations can enter fitting.')
    n = evidence['used']
    if type(n) is not int or not 8 <= n <= 32 or len(evidence['measurements']) != n:
        raise ValueError('Expected 8..32 already revealed measurements.')
    sigma = evidence['measurement_noise_sigma']
    if isinstance(sigma, bool) or not isinstance(sigma, (float, int)) or not math.isfinite(sigma) or sigma < 0:
        raise ValueError('Invalid known noise strength.')
    queries, observed = [], []
    for item in evidence['measurements']:
        query = item['query']
        if set(query) != {'local_spins', 'original_position'}:
            raise ValueError('Unexpected public query fields.')
        queries.append(decode_student_query({'local_spins': query['local_spins'], 'position': query['original_position']}))
        value = item['delta']
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Invalid observation.')
        observed.append(value)
    return queries, np.asarray(observed, dtype=float)


def fit_proposal(evidence, structures):
    """The 8..32-query equivalent of v3's fixed-32 numerical helper."""
    grammar = public_coverage()
    support = canonical_proposal(structures, grammar)
    queries, observed = visible_queries(evidence)
    known = np.asarray([base_flip(*q) for q in queries])
    residual = observed-known
    columns = [np.asarray([Expression((Term(s, 1.0),)).delta(*q) for q in queries])-known for s in support]
    design = np.column_stack(columns) if columns else np.empty((len(queries), 0))
    if columns:
        coefficients, _, rank, singular = np.linalg.lstsq(design, residual, rcond=None)
    else:
        coefficients, rank, singular = np.empty(0), 0, np.empty(0)
    expression = Expression(tuple(Term(s, float(k)) for s, k in zip(support, coefficients)))
    diagnostic = {'raw_structures': support, 'expression': expression.to_dict(),
                  'used': evidence['used'], 'design_rank': int(rank), 'columns': len(support),
                  'identifiable_given_proposal': int(rank) == len(support),
                  'singular_values': singular.tolist(),
                  'observation_rmse': float(np.sqrt(np.mean((design @ coefficients-residual)**2))),
                  'fitted_coefficients_before_zero_removal': coefficients.tolist(),
                  'zero_coefficient_structures_removed': len(support)-len(expression.terms),
                  'added_structures': 0, 'additional_measurements': 0}
    return expression, diagnostic


def signature(expression):
    return packed(expression.to_dict())


class StudentLaboratory:
    """Driver-side state; a language model receives JSON replies only.

    Do not hand this Python object to code-executing students. A remote worker
    must communicate through the serialized request/reply transport. This class
    is not a sandbox for arbitrary code running in the driver's process.
    """
    def __init__(self, lab, initial_queries):
        self.__lab = lab
        observed = lab.initial_observations(initial_queries)
        self.__evidence = {'measurement_noise_sigma': lab.noise_sigma, 'used': lab.used,
                           'measurements': [{'query': public_query(q), 'delta': value}
                                            for q, value in zip(initial_queries, observed)]}
        self.__pending = None
        self.__log = []
        self.__finished = False

    def evidence(self):
        return deepcopy(self.__evidence)

    def events(self):
        return deepcopy(self.__log)

    def fit(self, structures):
        if self.__finished:
            raise ValueError('Episode already finished.')
        _, diagnostic = fit_proposal(self.evidence(), structures)
        return deepcopy(diagnostic)

    def prepare(self, proposal):
        """Compute and lock exact predictions BEFORE any new measurement.

        Returns the committed numeric predictions so the caller can inspect them
        before reveal(ticket). Raw and fitted alternatives are both retained.
        Two raw supports that reduce to the same expression are NOT counted as a
        discriminating idea merely because their JSON differs.
        """
        if self.__finished or self.__pending is not None:
            raise ValueError('Finish/reveal state does not permit a new experiment.')
        if not isinstance(proposal, dict) or set(proposal) != {'hypotheses', 'queries', 'reason', 'update_condition'}:
            raise ValueError('Submit hypotheses, queries, reason and update_condition.')
        hypotheses = proposal['hypotheses']
        if not isinstance(hypotheses, list) or not 2 <= len(hypotheses) <= 3:
            raise ValueError('Submit a primary explanation and one or two alternatives.')
        raw_queries = proposal['queries']
        if not isinstance(raw_queries, list) or not raw_queries or len(raw_queries)+self.__lab.used > self.__lab.budget:
            raise ValueError('A nonempty batch must stay inside the remaining measurement budget.')
        if any(not isinstance(proposal[key], str) or not proposal[key].strip() for key in ('reason', 'update_condition')):
            raise ValueError('Reason and update condition must be nonempty strings.')
        queries = [decode_student_query(q) for q in raw_queries]
        models, fits, supports = [], [], []
        for hypothesis in hypotheses:
            if not isinstance(hypothesis, dict) or set(hypothesis) != {'structures'}:
                raise ValueError('Each hypothesis must give only its proposed structures.')
            model, diagnostic = fit_proposal(self.evidence(), hypothesis['structures'])
            models.append(model)
            fits.append(diagnostic)
            supports.append(diagnostic['raw_structures'])
        if len(set(supports)) != len(supports):
            raise ValueError('Alternatives must differ after structural canonicalization.')
        predictions = [[m.delta(*q) for q in queries] for m in models]
        record = {'committed_at_used': self.__lab.used, 'proposal': deepcopy(proposal),
                  'fits': fits, 'queries': [encode_query(q) for q in queries], 'predictions': predictions,
                  'distinct_fitted_hypotheses': len({signature(m) for m in models}),
                  'prediction_disagreement': bool(np.any(np.ptp(np.asarray(predictions), axis=0) > 1e-9))}
        record['before_sha256'] = hashlib.sha256(packed(record).encode('utf-8')).hexdigest()
        ticket = self.__lab.commit(queries, models, predictions, proposal['reason'], proposal['update_condition'])
        record['ticket'] = ticket
        self.__pending = (ticket, deepcopy(record), queries)
        self.__log.append({'kind': 'committed_before_observation', **deepcopy(record)})
        return deepcopy(record)

    def reveal(self, ticket):
        if self.__finished or self.__pending is None or ticket != self.__pending[0]:
            raise ValueError('Reveal requires the matching pending ticket.')
        _, record, queries = self.__pending
        observations = self.__lab.reveal(ticket)
        for query, value in zip(queries, observations):
            self.__evidence['measurements'].append({'query': public_query(query), 'delta': value})
        self.__evidence['used'] = self.__lab.used
        result = {'ticket': ticket, 'observations': observations, 'used': self.__lab.used,
                  'before_sha256': record['before_sha256']}
        self.__log.append({'kind': 'observations_revealed', **deepcopy(result)})
        self.__pending = None
        return deepcopy(result)

    def finish(self, structures):
        if self.__finished or self.__pending is not None:
            raise ValueError('Cannot finish twice or while a measurement is pending.')
        model, diagnostic = fit_proposal(self.evidence(), structures)
        result = {'used': self.__lab.used, 'raw_structures': deepcopy(structures),
                  'expression': model.to_dict(), 'fit': diagnostic}
        self.__log.append({'kind': 'finished', **deepcopy(result)})
        self.__finished = True
        return deepcopy(result)
