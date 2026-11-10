"""Conditional-on-seeds paired hierarchical bootstrap for the frozen endpoint."""
import numpy as np


def primary_comparison(rows, *, repetitions=10000, seed=98901, expected_counts=None):
    """Rows: group, seed, level, structure_id, world_id, sigma, mode, score.

    score is the prespecified tested_idea_recovery boolean. Mechanism structures
    and their parameter worlds are resampled; the three trained seeds remain
    fixed. Missing runs, duplicate cells, or disagreement about clusters fail.
    """
    selected = [r for r in rows if r['group'] in ('B','C') and r['mode']=='active'
                and r['sigma']==1 and r['level'] in ('combination','four_body')]
    cells, clusters = {}, {}
    for row in selected:
        key = (row['group'],row['seed'],row['world_id'])
        if key in cells or type(row['score']) is not bool:
            raise ValueError('Duplicate result or nonboolean primary endpoint.')
        cells[key] = int(row['score'])
        description = (row['level'],row['structure_id'])
        if row['world_id'] in clusters and clusters[row['world_id']] != description:
            raise ValueError('Pairing or mechanism cluster metadata changed.')
        clusters[row['world_id']] = description
    seeds = sorted({r['seed'] for r in selected})
    if seeds != [401,402,403]:
        raise ValueError('The registered three seeds are required.')
    for world in clusters:
        if any((group,s,world) not in cells for group in ('B','C') for s in seeds):
            raise ValueError('Incomplete paired grid; do not silently drop failures or absent jobs.')
    groups = {}
    expected_counts = expected_counts or {'combination':(50,11),'four_body':(50,50)}
    for level in ('combination','four_body'):
        structures = sorted({cluster for lev,cluster in clusters.values() if lev==level})
        if not structures:
            raise ValueError('Both structural holdout levels are required.')
        world_count=sum(lev==level for lev,_ in clusters.values())
        if (world_count,len(structures)) != tuple(expected_counts[level]):
            raise ValueError('Registered world/structure counts are incomplete; no whole-world omission is allowed.')
        groups[level] = []
        for structure in structures:
            worlds = sorted(w for w,c in clusters.items() if c==(level,structure))
            # Preserve B/C pairing and seed identity before averaging.
            difference = np.asarray([[cells['C',s,w]-cells['B',s,w] for w in worlds] for s in seeds],dtype=float)
            groups[level].append(difference)
    seed_effects = np.mean([np.stack([v.mean(axis=1) for v in values]).mean(axis=0) for values in groups.values()],axis=0)
    point = float(seed_effects.mean())
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions)
    for i in range(repetitions):
        levels = []
        for values in groups.values():
            sampled = []
            for index in rng.integers(len(values),size=len(values)):
                matrix = values[int(index)]
                worlds = rng.integers(matrix.shape[1],size=matrix.shape[1])
                sampled.append(matrix[:,worlds].mean())
            levels.append(np.mean(sampled))
        samples[i] = np.mean(levels)
    lower,upper = np.quantile(samples,[.025,.975],method='linear')
    return {'C_minus_B':point,'conditional_95_percentile_interval':[float(lower),float(upper)],
            'seed_effects':dict(zip(map(str,seeds),map(float,seed_effects))),
            'cluster_counts':{level:len(values) for level,values in groups.items()},
            'world_counts':{level:sum(v.shape[1] for v in values) for level,values in groups.items()},
            'bootstrap_repetitions':repetitions,'bootstrap_seed':seed,
            'consistent_process_advantage':bool(lower>0 and np.all(seed_effects>0)),
            'scope':'Conditional on these three trained seeds; not an equivalence test or a future-seed population interval.'}
