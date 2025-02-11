# takes a sequence of 'images' of generated pieces and outputs PDF file
import os

lilypond_path = 'lilypond-2.24.1/bin'
current_path = os.environ.get('PATH', '')
lilypond_environment_variable_path = f"{lilypond_path}:{current_path}"
os.environ['PATH'] = lilypond_environment_variable_path

import abjad
from music21 import *
import random
from fractions import Fraction
import numpy as np
import json
import Music_Generation as mg
import tensorflow as tf

# map encoded note durations to lilypond/abjad syntax
abj_duration_dict = {
    1: '16',
    2: '8',
    4: '4',
    8: '2',
    16: '1'
}

# process images ------------------------------------------------------------------------------------

# arranges notes into LH/RH, chords, voices
def arrange_notes(note_array, midi_offset):  # main function for music processing
    # separate notes into lh/rh
    lh_notes = []
    rh_notes = []
    for note in note_array:
        if note[2][0] >= 60:
        # midi offset is used if notes are encoded as midi numbers, not array index
            rh_notes.append(note)
        else:
            lh_notes.append(note)
    # merge notes into chords
    if len(lh_notes) > 0:
        lh_notes = find_chords(lh_notes)
    if len(rh_notes) > 0:
        rh_notes = find_chords(rh_notes)
    # extract voices
    lh_voices = []
    rh_voices = []
    if len(lh_notes) > 0:
        lh_voices = extract_voices(lh_notes)
    if len(rh_notes) > 0:
        rh_voices = extract_voices(rh_notes)
    return lh_voices, rh_voices

def find_chords(note_array):
# if any notes have the same step and duration, then they are merged into [step, duration, [pitches]]
    note_array = sorted(note_array, key=lambda x: x[0]) # sort notes by value of step
    # put notes with same step values in groups
    same_step_groups = [[]]
    same_step_groups[0].append(note_array[0])
    index = 0
    for i in range(len(note_array)-1):
        if note_array[i+1][0] == note_array[i][0]:
            same_step_groups[index].append(note_array[i+1])
        else:
            same_step_groups.append([note_array[i+1]])
            index += 1
    # combine notes in same group with same duration into a chord
    def insert(current_note, chord_notes_array):
        for c_note in chord_notes_array:
            if current_note[1] == c_note[1] and current_note[2][0] not in c_note[2]:
            # note has same duration and can be merged
                c_note[2].append(current_note[2][0])
            else:
            # note does not match existing durations and canot be merged
            # note is appended to array of new notes
                found = False
                for c in chord_notes_array:
                    if current_note[1] == c[1]:
                        found = True
                if found == False:
                    chord_notes_array.append(current_note)
        return chord_notes_array
    # merge notes with the same step and duration into chords
    new_notes = []
    for group in same_step_groups:
        chord_notes = [group[0]]
        for i in range(len(group)-1):
            chord_notes = insert(group[i+1], chord_notes)
        for note in chord_notes:
            new_notes.append(note)
    return new_notes

# takes notes in [step, duration, [pitches]] format and assigns notes into separate voices
def extract_voices(note_array):

    def add_voice(notes):
        voice = []
        current_step = 0 # step of most recent note to be placed into a voice
        for note in notes:
            if note[0] >= current_step:
                voice.append(note)
                current_step = note[0] + note[1] # this is note step + note duration
        voice = sorted(voice, key=lambda x: x[0])  # sort notes by value of step
        return voice

    notes_placed = 0 # number of notes assigned to voices
    voices = []
    note_array = sorted(note_array, key=lambda x: x[0]) # sort notes by value of step
    notes_to_place =  len(note_array)
    while notes_placed < notes_to_place:
        new_voice = add_voice(note_array)
        voices.append(new_voice)
        # remove notes once they have been assigned a voice
        for note in new_voice:
            note_array.remove(note)
        notes_placed += len(new_voice) # track no. of notes assigned to voice
    return voices

# write abjad score ------------------------------------------------------------------------------------

# rename the PDF file to specified name after it has been generated
def rename_abjad(filepaths_array, new_name):
    abj_dir = 'Data/Temporary Abjad Output/'
    for filepath in filepaths_array:
        if filepath[-3:] != 'pdf':
            os.remove(abj_dir + filepath)
        else:
            os.rename(abj_dir + filepath, abj_dir + new_name + '.pdf')
    return abj_dir + new_name + '.pdf'

# creates PDF score using Abjad/Lilypond
def assemble_score(music_sections, name, use_sharp_dict, key):
    # music_sections takes 'processed_sections' from Music_Generation_2
    # each section is one bar containing multiple voices
    # dictionary maps each key on the piano to Lilypond notation with sharps/flats
    if use_sharp_dict == True:
        f = open('Data/Abjad Sharp Notes.json', 'r')
        abj_notes_dict = json.load(f)
    else:
        f = open('Data/Abjad Flat Notes.json', 'r')
        abj_notes_dict = json.load(f)
    # initialise Abjad score
    rh_voice = abjad.Voice(name="RH_Voice")
    rh_staff = abjad.Staff([rh_voice], name="RH_Staff")
    lh_voice = abjad.Voice(name="LH_Voice")
    lh_staff = abjad.Staff(name="LH_Staff", simultaneous=True)
    lh_staff.extend([lh_voice])
    piano_staff = abjad.StaffGroup(lilypond_type="PianoStaff", name="Piano_Staff")
    piano_staff.extend([rh_staff, lh_staff])
    score = abjad.Score([piano_staff], name="Score")
    # process images for Lilypond
    lh_bars = []
    rh_bars = []
    for section in music_sections:
        bar_notes = arrange_notes(section, 0)
        lh_bars.append(bar_notes[0])
        rh_bars.append(bar_notes[1])
    # write notes as array of Abjad containers
    lh_container_array = []
    rh_container_array = []

    for bar in lh_bars:
        if len(bar) > 0: # bar contains notes
            lp_bar = write_abj_bar(bar, abj_notes_dict)
    # lp_bar looks like [['f,8', 'r8', 'ef8', 'r8', 'gf,8', 'r8 r8', 'bf,8'],
            #            ['af,4', 's16', 'af,16', 's8 s4 s4']]
            lh_container_array.append(write_containers(lp_bar))
        else: # bar contains no notes
            lh_container_array.append(abjad.Voice('r1'))
    for bar in rh_bars:
        if len(bar) > 0: # bar contains notes
            lp_bar = write_abj_bar(bar, abj_notes_dict)
            rh_container_array.append(write_containers(lp_bar))
        else: # bar contains no notes
            rh_container_array.append(abjad.Voice('r1'))
    # place containers into a voice
    for container in lh_container_array:
        lh_voice.append(container)
    for container in rh_container_array:
        rh_voice.append(container)
    # attach bass clef to first note or rest
    clef = abjad.Clef("bass")
    attach_to_score(lh_voice, clef, 0)
    # attach key signature to first note or rest
    with open('Data/Abjad Pitch Classes.json', 'r') as f:
        pitch_class_dict = json.load(f)
    key_signature = abjad.KeySignature(
        abjad.NamedPitchClass(pitch_class_dict[key][0]), abjad.Mode(pitch_class_dict[key][1]))
    attach_to_score(lh_voice, key_signature, 0)
    attach_to_score(rh_voice, key_signature, 0)
    # attach double bar line
    bar_line = abjad.BarLine("|.")
    attach_to_score(lh_voice, bar_line, -1)
    # generate PDF
    # Lilypond file settings
    preamble = rf"""#(set-global-staff-size 20)
    \header {{
    composer = \markup {{ GANdioso }}
    title = \markup {{ {name} }}
    }}

    \layout {{
        indent = 0
    }}"""
    lilypond_file = abjad.LilyPondFile([preamble, score])
    abjad.show(lilypond_file, output_directory='Data/Temporary Abjad Output', should_open=False)
    abj_path = rename_abjad(os.listdir('Data/Temporary Abjad Output'), name)
    f.close()
    return abj_path

# function to attach clef/key signature to score
def attach_to_score(voice, object, index):
    if isinstance(voice[index], abjad.Rest):
        abjad.attach(object, voice[index])
    elif isinstance(voice[index][index], abjad.Rest):
        abjad.attach(object, voice[index][index])
    else:
        abjad.attach(object, voice[index][index][index])

# place a bar's voices into a container
def write_containers(bar):
    # bar is an array of voices within a single bar
    # write new voices
    new_voices_array = []
    for voice in bar:
        voice_string = ""
        for note in voice:
            voice_string = voice_string + note + ' '
        voice_string = voice_string[:-1]
        new_voice = abjad.Voice(voice_string)
        new_voices_array.append(new_voice)
    container = abjad.Container(new_voices_array, simultaneous=True)
    return container

def write_abj_bar(bar_voices, abj_notes_dict):
    bar_voices = sorted(bar_voices, key=len)
    lp_bar_voices = []
    # longest voice written first as it is the only one with visible rests
    bar_voices[-1] = fill_in_rests(bar_voices[-1])
    lp_bar_voices.append(add_notes_to_voice(bar_voices[-1], True, abj_notes_dict))
    for i in range(len(bar_voices) - 1):
        new_voice = fill_in_rests(bar_voices[i])
        lp_bar_voices.append(add_notes_to_voice(new_voice, False, abj_notes_dict))
    # lp_bar_voices looks like[['ef8', 'ef16', 'gf16' ... ], [...] x number of voices]
    return lp_bar_voices

def add_notes_to_voice(voice_array, rests_visible, abj_notes_dict):
    # add notes to voices
    voice = []
    for note in voice_array:
        if note[2] == 0: # rest
            voice.append(write_rest(note, rests_visible))
        else: # note/chord
            voice.append(array_to_abj(note, abj_notes_dict))
    # voice looks like ['ef8', 'ef16', 'gf16', 'r8', 'r16' ... ]
    return voice

# writes rest in lilypond syntax: takes 1 rest only
def write_rest(rest_array, rests_visible):
    # file that maps rest intervals to Lilypond notation
    with open('Data/Abjad Rest Intervals.json', 'r') as g:
        abj_rest_dict = json.load(g)
        step = rest_array[0]
        duration = rest_array[1]
        if duration >= 2:
            rest_string = abj_rest_dict[f'{step} {step + duration - 1}']
        if duration <= 1:
            rest_string = 'r16'
        if rests_visible == False: # replace with invisible rests
            rest_string = rest_string.replace('r', 's')
        # returns something like 'r1 r2 r2 r2'
    return rest_string

# adds rests to a voice if notes are not being played
def fill_in_rests(voice):
    current_step = 0
    new_voice_array = []
    for note in voice:
        if note[0] > current_step:
            # add rest
            new_voice_array.append([current_step, note[0] - current_step, 0])
        new_voice_array.append(note)
        current_step = note[0] + note[1]  # set current step to end of note
        if current_step >= 16:
            break
    if current_step <= 15: # add extra rest if bar not yet filled up
        new_voice_array.append([current_step,  16 - current_step, 0])
    return new_voice_array

# turns notes from [step, duration, [pitches]] format into lilypond syntax
def array_to_abj(note, abj_notes_dict):
    lp_string = '' # lilypond string
    if len(note[2]) > 1: # chord
        for i in range(len(note[2])):
            lp_string = lp_string + abj_notes_dict[str(note[2][i])] + ' '
            # + 41 to map array index to midi number
        lp_string = f'<{lp_string}>{abj_duration_dict[note[1]]}' # note[1] is duration
    else: # single note
        lp_string = f'{abj_notes_dict[str(note[2][0])]}{abj_duration_dict[note[1]]}'
    # returns something like '<f bf >16' (chord) or 'gf16 (note)'
    return lp_string
