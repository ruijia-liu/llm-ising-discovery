"""Lossless, model-neutral compaction of the audited conversation messages.

Spin arrays become 64 hexadecimal characters; expressions omit the constant H0
label; repeated explanation strings use '=' to reference their previous value.
No observations, floating-point values, hypotheses, or messages are discarded.
"""
from copy import deepcopy
import json

PROSE_KEYS = {'meaning', 'reason', 'display_note', 'update_condition', 'explanation'}
CODEC_INSTRUCTION = ('\n存储格式：构型的 hex 是按行排列的256个自旋位，1表示+1，0表示-1，'
                     '共64个十六进制字符，position不变。表达式 {"terms":[[K,offsets],...]} '
                     '默认包含已知H0；空terms表示没有额外作用。说明文字字段的值"="表示沿用之前同名字段的文字。'
                     '数值均未舍入，完整实验记录没有截断。')


def _transform(value, prose, encode):
    if isinstance(value, list):
        return [_transform(v, prose, encode) for v in value]
    if not isinstance(value, dict):
        return value
    if encode and set(value) == {'rows', 'position'}:
        bits = ''.join(value['rows']).replace('+', '1').replace('-', '0')
        if len(bits) != 256 or set(bits)-set('01'):
            raise ValueError('Invalid spin array.')
        return {'hex': f'{int(bits, 2):064x}', 'position': value['position']}
    if not encode and set(value) == {'hex', 'position'}:
        if len(value['hex']) != 64:
            raise ValueError('Invalid hex configuration.')
        bits = f"{int(value['hex'], 16):0256b}".replace('1', '+').replace('0', '-')
        return {'rows': [bits[i:i+16] for i in range(0, 256, 16)], 'position': value['position']}
    if encode and set(value) == {'base', 'terms'}:
        if value['base'] != 'nearest_neighbour_J=1':
            raise ValueError('Unknown base.')
        return {'terms': [[t['coefficient'], t['offsets']] for t in value['terms']]}
    if not encode and set(value) == {'terms'}:
        return {'base': 'nearest_neighbour_J=1',
                'terms': [{'coefficient': t[0], 'offsets': t[1]} for t in value['terms']]}
    result = {}
    for key, child in value.items():
        if key in PROSE_KEYS and isinstance(child, str):
            if encode:
                if child == '=':
                    raise ValueError('Literal equals marker not supported in prose.')
                result[key] = '=' if prose.get(key) == child else child
                prose[key] = child
            else:
                if child == '=':
                    if key not in prose:
                        raise ValueError('Undefined prose reference.')
                    result[key] = prose[key]
                else:
                    result[key] = child
                    prose[key] = child
        else:
            result[key] = _transform(child, prose, encode)
    return result


def compact_messages(messages):
    result, prose = [], {}
    for message in messages:
        if message['role'] == 'system':
            content = message['content']+CODEC_INSTRUCTION
        else:
            content = json.dumps(_transform(json.loads(message['content']), prose, True),
                                 ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        result.append({'role': message['role'], 'content': content})
    return result


def expand_messages(messages):
    result, prose = [], {}
    for message in messages:
        if message['role'] == 'system':
            if not message['content'].endswith(CODEC_INSTRUCTION):
                raise ValueError('Missing codec instruction.')
            content = message['content'][:-len(CODEC_INSTRUCTION)]
        else:
            content = json.dumps(_transform(json.loads(message['content']), prose, False),
                                 ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
        result.append({'role': message['role'], 'content': content})
    return result


def normalized_messages(messages):
    result = deepcopy(messages)
    for message in result:
        if message['role'] != 'system':
            message['content'] = json.dumps(json.loads(message['content']), ensure_ascii=False, sort_keys=True,
                                            separators=(',', ':'), allow_nan=False)
    return result


def tokenize_assistant_only(tokenizer, messages):
    """Exact full-template tokenization; no truncation, only assistant loss labels."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    # This model's non-tool chat format is checked rather than assumed silently.
    pieces = []
    spans = []
    cursor = 0
    for message in messages:
        header = '<|im_start|>'+message['role']+'\n'
        body = message['content']+'<|im_end|>'
        pieces.append(header+body+'\n')
        if message['role'] == 'assistant':
            spans.append((cursor+len(header), cursor+len(header)+len(body)))
        cursor += len(header)+len(body)+1
    if ''.join(pieces) != text:
        raise ValueError('Chat template changed; re-audit assistant mask before training.')
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded['input_ids']
    labels = [-100]*len(ids)
    span_index = 0
    for j, (start, end) in enumerate(encoded['offset_mapping']):
        while span_index < len(spans) and start >= spans[span_index][1]:
            span_index += 1
        if span_index < len(spans) and spans[span_index][0] <= start < end <= spans[span_index][1]:
            labels[j] = ids[j]
    if not any(label != -100 for label in labels):
        raise ValueError('No assistant tokens found.')
    return ids, labels
