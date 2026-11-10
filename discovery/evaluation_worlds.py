"""Grader-only worlds and shared acquisition streams; formal access is gated."""
from pathlib import Path

import numpy as np

from build_publication_curriculum import read, sha
from .lab import Laboratory
from .physics import Expression, random_queries
from .student_laboratory import public_query


SOURCE=Path('data/research_curriculum_publication_v1')


def require_completed_main_training():
    protocol_hash=sha('MAINLINE_STUDY_PROTOCOL.md')
    for seed in (401,402,403):
        for group in 'ABC':
            directory=Path('results/student_mainline_v1')/f'{group}_seed{seed}'
            if not (directory/'result.json').exists():
                raise ValueError('Formal test access requires all nine completed registered training runs.')
            result,config=read(directory/'result.json'),read(directory/'config.json')
            if (not result['completed'] or result['completed_updates']!=1200 or result['group']!=group
                or result['seed']!=seed or config['sources']['MAINLINE_STUDY_PROTOCOL.md']!=protocol_hash
                or sha(directory/'adapter/adapter_model.safetensors')!=result['adapter_sha256']):
                raise ValueError('Training completion/provenance check failed.')


def load_worlds(scope):
    if scope=='development':
        return [{'world_id':w['world_id'],'world_index':w['world_index'],'level':'parameter_development',
                 'family':w['family'],'truth':w['truth']} for w in read(SOURCE/'private/worlds.json') if w['split']=='validation']
    if scope!='formal':
        raise ValueError('Unknown evaluation scope.')
    require_completed_main_training()  # Before opening any reserved answer file.
    path=SOURCE/'private/reserved_test_worlds.json'
    audit=read(SOURCE/'audit_report.json')
    if sha(path)!=audit['reserved_tests']['private_file_sha256']:
        raise ValueError('Reserved worlds changed since exposure audit.')
    return [{**w,'world_index':i} for i,w in enumerate(read(path)['worlds'])]


def streams(scope,world_index):
    seed={'development':98711,'formal':98731}[scope]
    def rng(purpose):
        return np.random.default_rng([seed,world_index,purpose])
    return {'initial':random_queries(rng(101),8),'fixed_additional':random_queries(rng(202),24),
            'noise_seed':np.random.SeedSequence([seed,world_index,404]),
            'search_seed':np.random.SeedSequence([seed,world_index,505]), 'evaluation_seed':seed}


def fixed_evidence(oracle,sigma,inputs):
    """Acquire common random observations once, with exactly 32 metered calls."""
    lab=Laboratory(oracle,noise_sigma=sigma,noise_seed=inputs['noise_seed'])
    queries=list(inputs['initial'])
    observations=lab.initial_observations(queries)
    extra=inputs['fixed_additional']
    base=Expression()
    ticket=lab.commit(extra,[base],[[base.delta(*q) for q in extra]],
                      'Prespecified random acquisition, independent of hidden world.',
                      'No adaptive query selection in fixed-evidence acquisition.')
    observations+=lab.reveal(ticket); queries+=extra
    evidence={'measurement_noise_sigma':sigma,'used':32,
              'measurements':[{'query':public_query(q),'delta':y} for q,y in zip(queries,observations)]}
    return evidence,lab.events


def prefix(evidence,budget):
    return {**evidence,'used':budget,'measurements':evidence['measurements'][:budget]}
