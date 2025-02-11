# takes a sequence of 'images' of generated pieces and outputs MIDI file
from music21 import *
import random
from fractions import Fraction
import numpy as np
import json
import PyPDF2
import Assemble_PDF as ap

# arranges notes into chords, voices
def arrange_notes(note_array):  # main function for music processing
    # merge notes into chords
    chord_notes = ap.find_chords(note_array)
    # extract voices
    all_voices = ap.extract_voices(chord_notes)
    return all_voices

# creates Midi file using Music21
def assemble_midi(music_sections, name, model):
    if model == 'Transformer':
        shift = 0
    else: shift =  40
    # map encoded pitches to midi numbers
    for a in range(len(music_sections)):
        for b in range(len(music_sections[a])):
            for c in range(len(music_sections[a][b][2])):
                music_sections[a][b][2][c] += shift
    # process music sections to extract voices (parts)
    all_bars = []
    for section in music_sections:
        if len(section) > 0:
            bar_notes = arrange_notes(section) # returns notes organised into voices
            # a bar is [[v1], [v2], ...]
            all_bars.append(bar_notes)
    # find the highest number of voices in a bar - this is no. of midi tracks needed
    num_voices = 0
    for bar in all_bars:
        if len(bar) > num_voices:
            num_voices = len(bar)
    # write notes as an array of 'measures', each measure containing a part for each voice
    all_measures = []
    for i in range(len(all_bars)):
        bar = all_bars[i]
        bar_array = [] # contains existing voices + new voices with rests
        if len(bar) >= 1:  # bar contains notes
            bar_parts = write_m21_bar(bar)
            for part in bar_parts:
                bar_array.append(part)
        for n in range(num_voices - len(bar)):
        # fill in remaining voices so that all bars have the same number of voices
            rest = note.Rest()
            rest.duration.quarterLength = 4
            bar_array.append([rest])
        all_measures.append(bar_array)
    # create Music21 score
    m21_main_stream = stream.Score()
    for i in range(num_voices):
        new_voice = stream.Part()
        for measure in all_measures:
            new_voice.append(measure[i]) # measure[i] is the ith voice in each bar
        m21_main_stream.append(new_voice)
    # generate MIDI file and save to directory
    midi_path = '/Users/danielwang/PycharmProjects/GANdioso_V2/Data/Temporary Midi Output/' + name + '.mid'
    m21_main_stream.write('midi', midi_path)
    return midi_path

# writes one bar as multiple parts, one for each voice
def write_m21_bar(bar_voices):
    bar_parts = []
    for voice in bar_voices:
        m21_part = write_voice(voice)
        bar_parts.append(m21_part)
    return bar_parts

# writes a single voice (format [step, duration, [pitches]] in m21 notation
def write_voice(voice):
    part = stream.Part()
    current_step = 0
    for note_step in voice:
        if note_step[0] > current_step: # if there is a gap between notes
            # add rest first
            rest = note.Rest()
            rest.duration.quarterLength = (note_step[0] - current_step)*0.25
            part.append(rest)
        # add note
        if len(note_step[2]) > 1:
            m21_note = chord.Chord(note_step[2])
        else:
            m21_note = note.Note(note_step[2][0])
        m21_note.duration.quarterLength = note_step[1]*0.25
        part.append(m21_note)
        current_step = note_step[0] + note_step[1]
    # add rest if bar not fully filled up yet
    if current_step < 16:
        rest = note.Rest()
        rest.duration.quarterLength = (16 - current_step) * 0.25
        part.append(rest)
    return part



