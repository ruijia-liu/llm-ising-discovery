"""Single-writer, durable receipts for exact replay of simulated episodes.

The oracle store is PRIVATE: it holds clean simulator outputs, never student
evidence. Laboratory reconstructs the original noisy observations from its
unchanged seeded stream. An unresolved reservation is never retried silently.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import uuid

from .physics import encode_query


class RecoveryError(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def read_checked(path):
    def unique_pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise RecoveryError('Duplicate key in a durable record.')
            result[key] = value
        return result
    value = json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique_pairs)
    canonical(value)  # Reject nonfinite values even if the JSON parser accepts them.
    return value


def durable_new(path, value):
    """Flush a complete new record before atomic rename; preserve orphan temps."""
    path = Path(path)
    if path.exists():
        raise RecoveryError('Do not overwrite a committed journal record.')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    with temporary.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(canonical(value)+'\n')
        handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)


class EpisodeJournal:
    def __init__(self, root, identity):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = (self.root/'.writer.lock').open('a+b')
        self._locked = False
        try:
            self._lock.seek(0, os.SEEK_END)
            if self._lock.tell() == 0:
                self._lock.write(b'\0'); self._lock.flush()
            self._lock.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self._lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._locked = True
        except OSError as error:
            self._lock.close()
            raise RecoveryError('Another live writer owns this episode; do not start a second run.') from error
        try:
            metadata = {'schema': 'durable-episode-v1', 'identity': identity,
                        'oracle_records_are_private_clean_simulator_values': True}
            path = self.root/'identity.json'
            if path.exists():
                if read_checked(path) != metadata:
                    raise RecoveryError('Episode identity/configuration changed; replay forbidden.')
            else:
                durable_new(path, metadata)
            self.requests, self.results, self.cursor = {}, {}, {'oracle': 0, 'model': 0, 'events': 0}
            self.replayed, self.created = {'oracle': 0, 'model': 0, 'events': 0}, {'oracle': 0, 'model': 0, 'events': 0}
            for kind in ('oracle', 'model'):
                folder = self.root/kind
                paths = sorted(folder.glob('*.request.json'))
                if [p.name for p in paths] != [f'{i:06d}.request.json' for i in range(len(paths))]:
                    raise RecoveryError('Noncontiguous call reservations; no silent omission.')
                self.requests[kind], self.results[kind] = [], []
                for index, request_path in enumerate(paths):
                    request = read_checked(request_path)
                    if request['index'] != index or request['request_sha256'] != digest(request['request']):
                        raise RecoveryError('Call reservation checksum/index mismatch.')
                    result_path = folder/f'{index:06d}.result.json'
                    if not result_path.exists():
                        raise RecoveryError(f'Unresolved {kind} call {index}: execution may have begun; no automatic replay of this call.')
                    result = read_checked(result_path)
                    if (result['request_sha256'] != request['request_sha256']
                            or result['value_sha256'] != digest(result['value'])):
                        raise RecoveryError('Call result checksum or request association changed.')
                    self.requests[kind].append(request)
                    self.results[kind].append(result)
                if len(list(folder.glob('*.result.json'))) != len(paths):
                    raise RecoveryError('Result without its call reservation.')
            paths = sorted((self.root/'events').glob('*.json'))
            if [p.name for p in paths] != [f'{i:06d}.json' for i in range(len(paths))]:
                raise RecoveryError('Noncontiguous protocol events.')
            self.events = []
            for index, event_path in enumerate(paths):
                row = read_checked(event_path)
                if row['index'] != index or row['event_sha256'] != digest(row['event']):
                    raise RecoveryError('Protocol event checksum/index mismatch.')
                self.events.append(row['event'])
            self.previous_completion = read_checked(self.root/'completion.json') if (self.root/'completion.json').exists() else None
        except Exception:
            self.close()
            raise

    def _call(self, kind, request, function):
        if not self._locked:
            raise RecoveryError('Journal is closed.')
        request = json.loads(canonical(request))
        index = self.cursor[kind]
        if index < len(self.requests[kind]):
            if self.requests[kind][index]['request'] != request:
                raise RecoveryError('Replayed call differs from its original query or prompt.')
            value = self.results[kind][index]['value']
            self.replayed[kind] += 1
        else:
            if self.cursor['events'] < len(self.events):
                raise RecoveryError('Cannot execute a new call before all saved protocol events have replayed.')
            if self.previous_completion is not None:
                raise RecoveryError('A completed episode cannot acquire additional calls.')
            # On any exception between these two durable writes, leave the
            # reservation unresolved. Retrying could conceal real extra cost.
            reservation = {'index': index, 'request': request, 'request_sha256': digest(request)}
            folder = self.root/kind
            durable_new(folder/f'{index:06d}.request.json', reservation)
            value = function()
            value = json.loads(canonical(value))
            result = {'request_sha256': reservation['request_sha256'], 'value': value, 'value_sha256': digest(value)}
            durable_new(folder/f'{index:06d}.result.json', result)
            self.requests[kind].append(reservation); self.results[kind].append(result)
            self.created[kind] += 1
        self.cursor[kind] += 1
        return deepcopy(value)

    def oracle(self, function):
        def measured(state, position):
            return self._call('oracle', encode_query((state, position)), lambda: float(function(state, position)))
        return measured

    def generator(self, function):
        def generate(messages, maximum):
            return self._call('model', {'messages': messages, 'maximum': maximum}, lambda: function(messages, maximum))
        return generate

    def event(self, event):
        if not self._locked:
            raise RecoveryError('Journal is closed.')
        event = json.loads(canonical(event))
        index = self.cursor['events']
        if index < len(self.events):
            if self.events[index] != event:
                raise RecoveryError('Protocol event diverged during replay; no modified continuation allowed.')
            self.replayed['events'] += 1
        else:
            if self.previous_completion is not None:
                raise RecoveryError('A completed episode cannot append protocol events.')
            durable_new(self.root/'events'/f'{index:06d}.json',
                        {'index': index, 'event': event, 'event_sha256': digest(event)})
            self.events.append(event)
            self.created['events'] += 1
        self.cursor['events'] += 1

    def complete(self, episode):
        if (self.cursor['oracle'] != len(self.requests['oracle']) or self.cursor['model'] != len(self.requests['model'])
                or self.cursor['events'] != len(self.events)):
            raise RecoveryError('Episode ended without replaying every recorded call/event.')
        value = {'episode_sha256': digest(episode), 'logical_oracle_calls': self.cursor['oracle'],
                 'logical_model_calls': self.cursor['model'], 'events': self.cursor['events']}
        if self.previous_completion is None:
            durable_new(self.root/'completion.json', value)
            self.previous_completion = value
        elif self.previous_completion != value:
            raise RecoveryError('Completed episode changed on reconstruction.')
        return {'schema': 'durable-episode-v1', 'logical_oracle_calls': self.cursor['oracle'],
                'logical_model_calls': self.cursor['model'], 'replayed_this_invocation': deepcopy(self.replayed),
                'created_this_invocation': deepcopy(self.created), 'ambiguous_calls_retried': 0}

    def close(self):
        if self._locked:
            self._lock.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self._lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._locked = False
        if not self._lock.closed:
            self._lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
