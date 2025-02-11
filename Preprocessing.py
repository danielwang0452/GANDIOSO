from music21 import *
from mido import MidiFile
import os
import ast
import fractions
import matplotlib.pyplot as plt
import numpy as np
from fractions import Fraction
import tensorflow as tf
import random

seq_length = 32 # 32 x 4 semiquavers = 2 4/4 bars
pitch_range = [40, 88]

# one-hot encode single midi file
def encode_midi(input_filepath, duration, note_range, output_filepath):
    # get loaded notes
    notes = load_from_midi(input_filepath)
    if notes != True:
        # turn note step, duration into number of semiquavers instead of number of crotchets
        for note in notes:
            note[0] /= 0.25
            note[1] /= 0.25
        # make 2D array for encoding piece - 32 arrays of 48 notes
        piece_array = []
        for i in range(int(notes[-1][0]) + int(notes[-1][1])): # step + duration of last note
            piece_array.append([])
            for j in range(48):
                piece_array[i].append(0)
        # insert notes into array
        for note in notes:
            insert(note, piece_array, note_range)
        # write encoded notes to GAN Data file
        with open(output_filepath, 'w') as g:
            for step in piece_array:
                g.write(str(step) + '\n')
        return piece_array
    else:
        return True


# inserts note into 2D one-hot encoded array by replacing 0s with 1s
def insert(note, array, note_range):
    # array is shape (duration length, pitches)
    # example note [11.0, 1.0, [63, 75]]
    step = note[0]
    duration = note[1]
    pitches = note[2]
    for pitch in pitches:
        if pitch >= note_range[0] and pitch < note_range[1]:
            for i in range(int(duration)):
                try:
                    array[int(step)+i-1][pitch-40] = 1
                except:
                    continue
    return array

# loads midi file into [step, duration, [pitches]]
def load_from_midi(filepath):
    notes = []
    try:
        midi_file = converter.parse(filepath)
    except:
        print(f'error with file {filepath}')
        return True
    # Iterate over all notes in all parts of the MIDI file
    for element in midi_file.flat.notesAndRests:
        # if midi element is a note:
        if isinstance(element, note.Note):
            step = element.offset
            duration = element.duration.quarterLength
            pitch = element.pitch.midi
            # convert to float if fractions are present
            if isinstance(element.duration.quarterLength, fractions.Fraction):
                duration = round(float(element.duration.quarterLength), 1)
            if isinstance(element.offset, fractions.Fraction):
                step = round(float(element.offset), 1)
            notes.append([step, duration, [pitch]])
            print([step, duration, [pitch]])
            step = round(element.offset * 4) / 4  # rounds to the nearest 0.25
        # if midi element is a chord
        elif isinstance(element, chord.Chord):
            step = element.offset
            duration = element.duration.quarterLength
            pitches = [n.pitch.midi for n in element]
            # convert to float if fractions are present
            if isinstance(element.duration.quarterLength, fractions.Fraction):
                duration = round(float(element.duration.quarterLength), 1)
            if isinstance(element.offset, fractions.Fraction):
                step = round(float(element.offset), 1)
            print([step, duration, pitches])
            step = round(element.offset * 4) / 4  # rounds to the nearest 0.25
            notes.append([step, duration, pitches])
    return notes

# turn a one-hot encdoded file into series of images
def encoded_to_image(filepath, steps_per_image):
    # load piece from text file
    with open(filepath, 'r') as f:
        piece_array = []
        for line in f:
            string = line[:-1]
            array = ast.literal_eval(string)
            piece_array.append(array)
    # turn piece into 2D array, or image
    for i in range(int((len(piece_array))/steps_per_image)):
        start_index = i*steps_per_image
        image = []
        for j in range(steps_per_image):
            image.append(piece_array[start_index + j])
        image = np.array(image)
        image = np.expand_dims(image, axis=-1)
        img = tf.keras.preprocessing.image.array_to_img(image)
        img.save(f'Encoded Images/piece_img_{i}.png')
    return

def preprocess_data():
    # gets all midi file paths for MAESTRO Dataset directory
    def load_maestro(directory):
        files_array = []
        for year in os.listdir(directory):
            if year != '.DS_Store':
                for path in os.listdir(f'{directory}/{year}'):
                    files_array.append(f'{directory}/{year}/{path}')
        return files_array
    # gets all midi file paths for Classical Piano Midi directory
    def load_filepaths(directory):
        files_array = []
        for piece in os.listdir(directory):
            if piece != '.DS_Store':
                files_array.append(f'{directory}/{piece}')
        return files_array
    filepath_array = load_maestro('Maestro Dataset') + load_filepaths('Classical Piano Midi')  # load dataset filepaths
    midi_notes = []
    encoded_pieces = []
    count = 0
    for piece_path in filepath_array[:]:
        count += 1
        print(f'{count}/{len(filepath_array)}:{str(piece_path)}')
        encoded_midi = encode_midi(piece_path, seq_length, pitch_range, 'GAN Text Data/Single Piece Encoded')
        if encode_midi(piece_path, seq_length, pitch_range, 'GAN Text Data/Single Piece Encoded 2') != True:
            encoded_pieces.append(encoded_midi)
            print(len(encoded_pieces[-1]))
        else:
            continue
    # write all notes to text file
    with open('GAN Text Data/GAN Data', 'w') as g:
        for piece in encoded_pieces:
            for step in piece:
                g.write(str(step) + '\n')
    return

# gets a random section from encoded file
def generate_test_piece(bars, lines_to_load):
    # load in notes from test data up to lines_to_load
    steps = []
    with open('GAN Text Data/GAN Data', 'r') as f:
        count = 0
        for line in f:
            count += 1
            if count > lines_to_load:
                break
            note = ast.literal_eval(line[:-1])
            steps.append(note)
    # write loaded notes to new file
    num_steps = bars*16
    start_step = random.randint(0, lines_to_load-num_steps)
    with open('GAN Text Data/Test Piece', 'w') as f:
        for i in range(num_steps):
            f.write(str(steps[start_step+i]) + '\n')

#preprocess_data()
#encode_midi(filepath, seq_length, pitch_range, 'GAN Text Data/Single Piece Encoded')






