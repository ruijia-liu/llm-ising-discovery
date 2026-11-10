"""Exact public-grammar algebra audit; no worlds, training labels or test data."""
from collections import Counter

import numpy as np

from build_publication_curriculum import atomic_json,sha
from discovery.physics import Grammar,random_queries


def anchored_monomials(structure):
    result=[]
    for anchor in structure:
        points=tuple(sorted(((p[0]-anchor[0])%16,(p[1]-anchor[1])%16) for p in structure))
        result.append(points)
    return result


def run():
    grammar=Grammar()
    owners={}; monomials=[]
    for index,structure in enumerate(grammar.structures):
        column=anchored_monomials(structure)
        if len(set(column))!=len(structure):
            raise AssertionError('A nontrivial translation stabilizer changes the norm formula.')
        for monomial in column:
            if (0,0) not in monomial or monomial in owners:
                raise AssertionError('Two canonical structures share an anchored spin monomial.')
            centred=[((x+8)%16-8,(y+8)%16-8) for x,y in monomial]
            if any(abs(x)>2 or abs(y)>2 for x,y in centred):
                raise AssertionError('A term depends on spins outside the public 5x5 patch.')
            owners[monomial]=index
        monomials.append(column)
    queries=random_queries(np.random.default_rng(99001),64)
    actual=grammar.features([q[0] for q in queries],[q[1] for q in queries])
    explicit=np.empty_like(actual)
    for i,(state,position) in enumerate(queries):
        for j,column in enumerate(monomials):
            explicit[i,j]=2*sum(np.prod([state[(position[0]+x)%16,(position[1]+y)%16] for x,y in monomial]) for monomial in column)
    np.testing.assert_array_equal(actual,explicit)
    # Distinct Walsh monomials are orthogonal under all independent uniform spins.
    # Disjoint monomial families therefore give the exact population Gram matrix.
    gram=np.diag([4*len(s) for s in grammar.structures])
    result={'structures':len(grammar.structures),'structures_by_order':dict(Counter(map(len,grammar.structures))),
            'distinct_anchored_monomials':len(owners),'off_diagonal_gram_entries_nonzero':0,
            'gram_diagonal_by_order':{k:4*k for k in range(1,5)},
            'population_gram_full_rank':int(np.linalg.matrix_rank(gram))==158,
            'all_terms_fit_5x5_patch':True,'runtime_feature_checks':int(actual.size),
            'finite_32_query_guarantee':False,'formal_tests_used':False,
            'source_sha256':{'audit_identifiability.py':sha('audit_identifiability.py'),'discovery/physics.py':sha('discovery/physics.py')}}
    atomic_json('results/identifiability_audit/audit.json',result)
    print(result)


if __name__=='__main__':
    run()
