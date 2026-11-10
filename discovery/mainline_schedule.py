"""Pair evidence/answer exposure and approximately match B/C input tokens."""
from collections import Counter

import numpy as np


def paired_schedule(records, seed, passes):
    if type(passes) is not int or passes < 1 or not records:
        raise ValueError('A nonempty fixed-pass schedule is required.')
    if len({row['episode_id'] for row in records}) != len(records):
        raise ValueError('Duplicate curriculum episode ID.')
    if any(row['auxiliary'][g]['input_tokens'] <= 0 for row in records for g in 'BC'):
        raise ValueError('Invalid auxiliary token length.')
    rng = np.random.default_rng(seed)
    steps, b_tokens, c_tokens = [], 0, 0
    for epoch in range(passes):
        for index in rng.permutation(len(records)):
            row = records[int(index)]
            b_length, c_length = (row['auxiliary'][g]['input_tokens'] for g in 'BC')
            # Match cumulative input volume using repetitions of exactly this
            # paired episode. Never repeat a terminal answer for token matching.
            c_tokens += c_length
            repetitions = max(1, int(np.floor((c_tokens-b_tokens)/b_length+0.5)))
            b_tokens += repetitions*b_length
            steps.append({'pass': epoch+1, 'episode_id': row['episode_id'],
                          'B_auxiliary_repetitions': repetitions, 'C_auxiliary_repetitions': 1,
                          'B_C_terminal_prefixes': [8, 16, 24, 32],
                          'A_terminal_prefixes': [8, 8, 8, 8]})
    relative = (b_tokens-c_tokens)/c_tokens
    if abs(relative) > 0.01:
        raise ValueError('Cannot match this corpus to 1% with whole-episode repetitions.')
    counts = Counter(row['episode_id'] for row in steps)
    if set(counts.values()) != {passes}:
        raise AssertionError('Paired episode presentations differ.')
    return {'steps': steps, 'updates_per_arm': len(steps),
            'terminal_presentations_per_arm': len(steps)*4,
            'auxiliary_input_tokens_B': b_tokens, 'auxiliary_input_tokens_C': c_tokens,
            'relative_auxiliary_input_difference': relative,
            'matched_terminal_weights_B_C': True, 'A_auxiliary_tasks': 0,
            'note': 'A is an answer-only auxiliary comparison with less input compute; primary inference is C versus B.'}
