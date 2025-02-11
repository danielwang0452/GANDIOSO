import torch
from torch import nn
import Transformer_Model as tm
import json
import numpy as np
import Transformer_Decode_Outputs as tdo
import sys
import Assemble_PDF as ap
import Assemble_Midi as am
import Music_Generation as mg
import Transformer_Assemble_PDF as tap
import random

device = 'mps'

VOCAB_SIZE = 417
SEQ_LENGTH = 512
model = tm.TransformerModel(ntoken=VOCAB_SIZE, d_model=512, nhead=4, d_hid=1024, nlayers=4, dropout=0.1)
model.to(device)
state_dict = torch.load('Torch_Experiments/Models/Transformer_1/epoch118_sd.pth', map_location='mps')
model.load_state_dict(state_dict)
softmax = nn.Softmax(dim=2)

def transformer_generate(num_bars, title, target_key):
    '''
    :param num_bars: 50
    :param title: 'Transformer_piece_1'
    :return:
    '''
    all_notes = []
    time_threshold = num_bars * 16 #
    time_elapsed = 0
    seed = None
    #_, _ = transformer_sample(seed, num_samples=64) # if we sample n tokens,
    # the next time we sample, the first n tokens will be very fast; from n
    # onwards, it will be back to normal speed
    while time_elapsed < time_threshold:
        sample_array, tokens = transformer_sample(seed, num_samples=128)
        seed = np.array(tokens[-20:]) - 1
        seed = seed.tolist()
        # add time elapsed to new notes
        for note in sample_array:
            note[0] += time_elapsed
        time_elapsed = sample_array[-1][0] # offset of last note\
        all_notes += sample_array
    # trim notes to match num bars
    for n, note in enumerate(all_notes):
        if note[0] + note[1] > time_threshold:
            break
    all_notes = all_notes[:n]
    all_notes = shift_notes(all_notes, target_key)
    # assemble midi file
    midi_path = am.assemble_midi([all_notes],
                                 title,
                                 model='Transformer')
    # process transformer output before generating PDF
    new_notes = []
    for note in all_notes:
        if len(note[2]) > 1:
            for pitch in note[2]:
                new_notes.append([note[0], note[1], [pitch]])
        else:
            new_notes.append(note)
    music_sections, key = process_score_notes(new_notes)
    # assemble PDF file
    use_sharp_dict = False
    if key in ["C Major", "G Major", "D Major", "A Major", "E Major", "B Major", "A Minor",
        "E Minor", "B Minor", "F# Minor", "C# Minor", "G# Minor"]:
        use_sharp_dict = True
    abj_path = tap.assemble_score(music_sections, title, use_sharp_dict, key)
    return midi_path, abj_path

def shift_notes(notes, target_key):
    '''
    :param notes: all generated notes
    :param target_key: e.g. 'C Major'
    :return: pitches all shifted by (target_key - current_key)
    '''
    # get current key
    current_key = find_key(notes)
    offset_dict = {
        'C Major': 0,
        'C Minor': 0,
        'C# Minor': 1,
        'Db Major': 1,
        'D Major': 2,
        'D Minor': 2,
        'Eb Major': 3,
        'Eb Minor': 3,
        'E Major': 4,
        'E Minor': 4,
        'F Major': 5,
        'F Minor': 5,
        'Gb Major': 6,
        'F# Minor': 6,
        'G Major': 7,
        'G Minor': 7,
        'Ab Major': 8,
        'G# Minor': 8,
        'A Major': 9 ,
        'A Minor': 9,
        'Bb Major': 10,
        'Bb Minor': 10,
        'B Major': 11,
        'B Minor': 11
    }
    difference = offset_dict[target_key]-offset_dict[current_key]
    if difference > 5:
        difference -= 12
    if difference < -5:
        difference += 12
    for note in notes:
        note[2][0] += difference
    # remove any notes that have gone out of piano range
    new_notes = []
    for note in notes:
        if note[2][0] < 21 or note[2][0] > 108:
            pass
        else:
            new_notes.append(note)
    return new_notes

def find_key(notes):
    '''
    :param notes: all notes in the piece [4, 56, 78, ... ]
    :return scale: 'C Major' for example
    '''
    with open('Data/Scale Pitches.json', 'r') as f:
        scale_dict = json.load(f)
        scale_count = {}
        # initialise count dict
        for key, val in scale_dict.items():
            scale_count[key] = 0
        # accumulate count dict
        for scale, pitches in scale_dict.items():
            for note in notes:
                if note[2][0] in pitches:
                    scale_count[scale] += 1
        # get most likely key
        max = 0
        max_key = None
        for scale, count in scale_count.items():
            if count > max:
                max_key = scale
                max = count
    return max_key

def process_score_notes(notes):
    '''
    :param notes: all notes in the piece [4, 56, 78, ... ]
    :return bars: notes split into num_bars [[4, 56, 78], [], ... ]
    :return key: 'C Major' for example
    '''
    # shift note offsets by 1 so it starts from 0
    for note in notes:
        note[0] -= 1
    # 1. find the key that the generated sequence is in
    key = find_key(notes)
    # 3. separate notes into bars
    num_bars = notes[-1][0] // 16 + 1
    bars = []
    for i in range(num_bars):
        bar = []
        lower_bound = i*16 # time offset bounds for current bar
        upper_bound = lower_bound + 15
        for note in notes:
            if note[0] >= lower_bound and note[0] <= upper_bound:
                if note[0] + note[1] <= upper_bound:
                    bar.append(note)
                elif note[0] + note[1] - 1 > upper_bound:
                    # if note crosses barline, split it into two notes
                    current_note = [note[0], upper_bound-note[0]+1, note[2]]
                    bar.append(current_note)
                    new_note = [upper_bound+1, note[0]+note[1]-upper_bound-1, note[2]]
                    notes.append(new_note)
        # trim notes durations (start on quaver, duration=2^n semiquavers)
        bar, _ = mg.trim_durations(bar, requires_image=False)
        bars.append(bar)
    # shift offset so that each bar starts at offset = 0
    for b, bar in enumerate(bars):
        for note in bar:
            note[0] -= b*16
        bar = sorted(bar, key=lambda x: x[0])
    return bars, key

def transformer_sample(seed, num_samples):
    '''
    :param seed: e.g. [4, 56, 78]
    :param num_samples: 256
    :return: outputs [4, 56, 78, ... num_samples]
    '''
    if seed == None:
        seed = [VOCAB_SIZE - 1] # start token
    inputs = [seed]
    inputs[0][0] = VOCAB_SIZE - 1 # setting start token is critical
    outputs = []
    model.eval()
    with torch.no_grad():
        for i in range(num_samples - len(seed) + 1):
            sample_index = len(inputs[0]) - 1
            pred_logits = model(torch.tensor(inputs, dtype=torch.long, device=device))
            pred_probs = softmax(pred_logits)
            pred_index = torch.multinomial(pred_probs[0, sample_index], num_samples=1, replacement=True)
            inputs[0].append(pred_index)
            outputs.append(int(pred_index) + 1)
            print(i)
    array = tdo.decode_outputs(outputs)
    return array, outputs


