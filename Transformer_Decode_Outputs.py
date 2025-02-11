import torch
import numpy as np
import json

'''
takes data in format [2, 232, 208, 50, 285, 198, 111 ...] 
and converts it to [step, duration, [pitches]] format
'''

def numbers_to_tokens(sequence):
    '''
    :param sequence: [2, 232, 208, 50, 285, 198, 111 ...]
    :return:
    '''
    token_sequence = []
    with open('Torch_Experiments/inverted_token_map.json', 'r') as f:
        token_map = json.load(f)
        for num in sequence:
            if num <= 415: # 417 is vocab size excluding padding and start tokens
                token_sequence.append(token_map[f'{num}'])
    return token_sequence

def validate_tokens(token_sequence):
    '''
    ensures that sequences follow these rules | and if not:
    1. first token is n{} | remove the first tokens until satisfied
    2. n{} is always followed by d{} | remove next non-d{} tokens
    3. d{} is followed by n{} or t{} i.e. no consecutive d{}'s | remove additional d{}'s
    4. t{} is followed by n{} | remove non-n{} tokens
    5. the last token is d{} | remove last tokens until satisfied

    rules applied in order 1, 5, (2, 3, 4)
    note (2, 3, 4) order may not be optimal - if there even is an optimal ordering
    '''
    def rule_1(sequence):
        a = len(sequence)
        while sequence[0][0] != 'n':
            del sequence[0]
        return sequence, a - len(sequence)

    def rule_2(sequence):
        a = len(sequence)
        i = 0
        while i+1 < len(sequence):
            if sequence[i][0] == 'n' and sequence[i+1][0] != 'd':
                del sequence[i+1]
                i -= 1
            i += 1
        return sequence, a - len(sequence)

    def rule_3(sequence):
        a = len(sequence)
        i = 0
        while i+1 < len(sequence):
            if sequence[i][0] == 'd' and sequence[i+1][0] == 'd':
                del sequence[i+1]
                i -= 1
            i += 1
        return sequence, a - len(sequence)

    def rule_4(sequence):
        a = len(sequence)
        i = 0
        while i+1 < len(sequence):
            if sequence[i][0] == 't' and sequence[i+1][0] != 'n':
                del sequence[i+1]
                i -= 1
            i += 1
        return sequence, a - len(sequence)

    def rule_5(sequence):
        a = len(sequence)
        while sequence[-1][0] != 'd':
            del sequence[-1]
        return sequence, a - len(sequence)
    # apply rules
    token_sequence, r1 = rule_1(token_sequence)
    token_sequence, r2 = rule_5(token_sequence)
    token_sequence, r3 = rule_2(token_sequence)
    token_sequence, r4 = rule_3(token_sequence)
    token_sequence, r5 = rule_4(token_sequence)
    # r{n} tracks how many token were removed by rule{n}
    print(r1, r2, r3, r4, r5)
    return token_sequence

def tokens_to_array(tokens):
    '''
    turns tokens in ['n45', 'd8', 't2' ... ] format into [step, duration, [pitch]] format
    '''
    array = []
    current_step = 1
    for t, token in enumerate(tokens):
        if token[0] == 'n':
            new_note = [current_step, 0, [int(token[1:])]]
        elif token[0] == 'd':
            new_note[1] = int(token[1:])
            array.append(new_note)
        elif token[0] == 't':
            current_step += int(token[1:])
    return array

# master function
def decode_outputs(sequence):
    '''
    :param sequence:  [2, 232, 208, 50, 285, 198, 111 ...]
    :return: [step, duration, [pitches]]
    '''
    tokens = numbers_to_tokens(sequence)
    validated_tokens = validate_tokens(tokens)
    array_notes = tokens_to_array(validated_tokens)
    return array_notes



