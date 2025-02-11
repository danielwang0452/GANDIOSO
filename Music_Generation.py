# this module processes the GAN generated images into numerically encoded notes
import os
import sys
import time
from music21 import *
import abjad
import random
import pygame
import subprocess
from fractions import Fraction
import ast
import json
import numpy as np
import tensorflow as tf
pitch_range = [0, 48]

# coverts image of music encoded as 2D array of 0s and 1s
# to encoding format where each note is [step, duration, [pitch/es]]
def image_to_array(image): # image has shape (no. of steps, no. of pitches)

    # function to extract a single note into [step, duration, [pitches]] format
    # given a pixel with a note in 2D encoded array
    def add_note(image, step, pitch):
        note = [step, 1, [pitch]]
        if step == 0:  # if input pixel is first step of bar
            i = 1
            while image[step+i][pitch] > 0.5:
                note[1] += 1 # add 1 to duration if note is held in next time
                if step+i >= 15:
                    break
                i += 1
        if step > 0:
            if image[step-1][pitch] > 0.5: # check if the input pixel is the first pixel of a note
                return None
            else:
                i = 1
                while step+i < len(image): # make sure doesn't go past image edge
                    if image[step+i][pitch] > 0.5: # if next pixel in 'step' direction is played
                        note[1] += 1
                        i += 1
                    else:
                        i = len(image)
        return note # note is in [step, duration, [pitches]]

    steps = len(image)
    num_pitches = len(image[0])
    notes = [] # notes will be appended to this in [step, duration, [pitches]] format
    for step in range(steps):
        for pitch in range(num_pitches):
            if image[step][pitch] > 0.5:
                current_note = add_note(image, step, pitch)
                if current_note == None:
                    continue
                else:
                    notes.append(current_note)
    return notes

# converts notes from [step, duration, [pitches]] format
# to 2D array of 0s and 1s format
def array_to_image(notes, num_steps, pitch_range):

    # inserts note into 2D one-hot encoded array by replacing 0s with 1s
    def insert(note, array, note_range):
        # array is shape (n, 48) i.e (duration length, pitches)
        # example note [11.0, 1.0, [63, 75]]
        step = note[0]
        duration = note[1]
        pitches = note[2]
        for pitch in pitches:
            if pitch >= note_range[0] and pitch < note_range[1]:
                for i in range(int(duration)):
                    try:
                        array[int(step) + i][pitch] = 1
                    except:
                        continue
        return array

    # create 2D array of 0s as the image
    image = []
    for i in range(num_steps):
        step = []
        for j in range(pitch_range[1]-pitch_range[0]):
            step.append(0)
        image.append(step)
    # add notes to images
    for note in notes:
        image = insert(note, image, pitch_range)
    return image

def trim_durations(note_array, requires_image):
# shortens notes so that they start/end on quavers
# image_array is 2D array of 0s and 1s
# note_array has same notes as image_array but in [step, duration, [pitches]] format
    valid_durations = [1, 2, 4, 8, 16]
    for note in note_array:
        if note[1] > 1: # duration longer than 1 semiquaver
            # shift start of note
            if note[0] % 2 != 0: # step not on start of quaver
                note[0] += 1
                note[1] -= 1
            # choose a valid duration for each note
            i = 0
            while valid_durations[i] <= note[1]:
                if i >= 4:
                    i = 5
                    break
                i += 1
            note[1] = valid_durations[i-1]
    if requires_image == True:
        trimmed_image = array_to_image(note_array, 16, [0, 48])
    else:
        trimmed_image = None
    return note_array, trimmed_image

def force_pitches(combined_sequence, target_key, non_conversion_percentage):
    # non_conversion_percentage is percentage of out of key notes to NOT convert
    new_sequence = []

    with open('Data/Scale Pitches.json', 'r') as f:
        key_dict = json.load(f)
        valid_pitches = key_dict[target_key]
        # for each step/chord
        for step in combined_sequence:
            new_step = [step[0], step[1], []]
            # for each pitch in the step/chord
            for pitch in step[2]:
                if random.randint(0,100) >= non_conversion_percentage:
                    # choose closest valid pitch
                    new_pitch = min(valid_pitches, key=lambda x: abs(x - (pitch + 40)))
                    new_step[2].append(new_pitch - 40)
                    # +- 40 to map 2D array indices to midi numbers
                else:
                    new_step[2].append(pitch)
            new_sequence.append(new_step)
    return new_sequence

def process_generated_images(image, scale, dissonance_value):
    note_array = image_to_array(image)
    trimmed_image = trim_durations(note_array, requires_image=True)[0]
    notes_in_key = force_pitches(trimmed_image, scale, dissonance_value)
    notes_in_key = trim_durations(notes_in_key, requires_image=True)[0] # trim durations again as forcing pitches creates messy durations
    processed_image = array_to_image(notes_in_key, 16, [0, 48])
    processed_image = trim_durations(image_to_array(processed_image), requires_image=True)[1]
    return processed_image

def GAN_generate(num_images, num_steps, vocab_size, index_range):
    generator = tf.keras.models.load_model('Data/GAN.h5')
    # shape of images array: (num_images, num_steps, vocab_size, 1)
    # get random vector to generate piece
    vector_seed = np.random.randint(1000)
    if random.randint(1,2) == 1:
        vector = tf.random.uniform((num_images, 128, 1), seed=vector_seed)
    else:
        vector = tf.random.normal((num_images, 128, 1), seed=vector_seed)
    # generate piece and reshape GAN output tensor
    imgs = generator.predict(vector)
    imgs = imgs.reshape(num_images, num_steps, vocab_size)
    imgs = imgs.tolist()

    def image_preview(image):
        image = np.array(image)
        image = np.expand_dims(image, axis=-1)
        img = tf.keras.preprocessing.image.array_to_img(image)
        img.save(f'GAN Setup/Test Image.png')

    return imgs

def generate_music(num_bars, scale, dissonance_value, note_density):
    # generate piece with GAN
    sections = GAN_generate(num_bars*5, 32, 48, [0, 48])
    # process each GAN image/section
    processed_sections = []
    for section in sections:
        bar1_image = process_generated_images(section[0:16], scale, dissonance_value)
        bar1 = image_to_array(bar1_image)
        bar2_image = process_generated_images(section[16:32], scale, dissonance_value)
        bar2 = image_to_array(bar2_image)
        processed_sections.append(bar1) # 1st bar
        processed_sections.append(bar2) # 2nd bar
    processed_sections = select_sections(processed_sections, num_bars, note_density)
    # remove empty bars
    for section in processed_sections:
        if len(section) == 0:
            processed_sections.remove(section)
    return processed_sections

def insert(note, array, note_range):
    # array is shape (n, 48) i.e (duration length, pitches)
    # example note [11.0, 1.0, [63, 75]]
    step = note[0]
    duration = note[1]
    pitches = note[2]
    for pitch in pitches:
        if pitch >= note_range[0] and pitch < note_range[1]:
            for i in range(int(duration)):
                try:
                    array[int(step) + i][pitch] = 1
                except:
                    continue
    return array

# function to select GAN images to include in the generated piece based on specified note density
def select_sections(all_sections, num_bars, note_density):
    # remove empty sections
    for section in all_sections:
        if len(section) == 0:
            all_sections.remove(section)
    # sort sections based on number of notes in each section
    all_sections = sorted(all_sections, key=len)
    index = int(0.5*num_bars) + 0.01*note_density*(len(all_sections)-num_bars)
    # get sections of specified note density
    chosen_sections = all_sections[int(index-0.5*num_bars):int(index+0.5*num_bars)]
    random.shuffle(chosen_sections)
    return chosen_sections[:num_bars]

