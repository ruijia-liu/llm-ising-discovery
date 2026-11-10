"""Resolve registered final adapters without exposing any heldout world."""
from pathlib import Path

from build_publication_curriculum import read, sha


def registered_student(study, group, seed):
    if seed not in (401, 402, 403):
        raise ValueError('Use a registered training seed.')
    if study == 'mainline' and group in ('A', 'B', 'C'):
        root = Path('results/student_mainline_v1')
        protocol = 'MAINLINE_STUDY_PROTOCOL.md'
        expected_stage = 'formal_mainline_v1'
    elif study == 'prediction_ablation' and group == 'C_no_predictions':
        root = Path('results/student_process_ablation_v1')
        protocol = 'PROCESS_ABLATION_PROTOCOL.md'
        expected_stage = 'formal_process_ablation_v1'
    else:
        raise ValueError('This study/group combination is not registered.')
    directory = root/f'{group}_seed{seed}'
    if not (directory/'result.json').exists():
        raise ValueError('Evaluate only the completed final 1200-step adapter.')
    result, config = read(directory/'result.json'), read(directory/'config.json')
    if (not result['completed'] or result['completed_updates'] != 1200 or result['group'] != group
            or result['seed'] != seed or config['stage'] != expected_stage or config['group'] != group
            or config['seed'] != seed or config['maximum_updates'] != 1200
            or config['sources'][protocol] != sha(protocol)):
        raise ValueError('Final training provenance or protocol does not match.')
    adapter = directory/'adapter'
    if sha(adapter/'adapter_model.safetensors') != result['adapter_sha256']:
        raise ValueError('Final adapter differs from its completed training record.')
    model = Path('models/Qwen2.5-0.5B-Instruct')
    if Path(config['model']) != model or config['model_revision'] != '7ae557604adf67be50417f59c2c2f167def9a775':
        raise ValueError('Unexpected base model for this registered profile.')
    for name, expected in config['base_files_sha256'].items():
        if sha(model/name) != expected:
            raise ValueError('Pinned inference model/tokenizer changed after training.')
    return {'study': study, 'model': model, 'adapter': adapter, 'training_config_sha256': sha(directory/'config.json'),
            'training_result_sha256': sha(directory/'result.json'), 'adapter_sha256': result['adapter_sha256'],
            'model_revision': config['model_revision'], 'protocol': protocol}
