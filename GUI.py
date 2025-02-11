# parts of the code in this file were adapted from the "PyQt5 GUI Thursdays" tutorial playlist by John Elder from
# Codemy.com, https://www.youtube.com/playlist?list=PLCC34OHNcOtpmCA8s_dpPMvQLyHbvxocY.

# this module contains everything related to the GUI

import os
from PyQt6 import QtWidgets, uic, QtGui, QtCore
import sys
import time
from music21 import *
import random
import shutil
import pygame
import subprocess
from fractions import Fraction
import ast
import Music_Generation as mg
import Assemble_PDF as ap
import Assemble_Midi as am
import Transformer_Sampling as ts
import fitz
import numpy as np
import mido
import json
import PyPDF2

# this class contains everything related to the GUI
class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        ui_file = os.path.abspath('Data/PyQtGUI.ui')
        uic.loadUi(ui_file, self)
        self.setWindowTitle('GANdioso')
        # get screen size and set window to that size
        screen = QtGui.QGuiApplication.primaryScreen()
        geometry = screen.geometry()
        self.setGeometry(geometry)
        self.initialise_parameters()  # load parameters from Data/Parameters.json
        # Generate Tab-------------------------------------------------------------------------------------
        # define and initialise Generate Tab widgets
        self.generate_info_button = self.findChild(QtWidgets.QPushButton, 'generate_info_button')
        self.numbars_spinbox = self.findChild(QtWidgets.QSpinBox, 'numbars_spinbox')
        self.numbars_spinbox.setValue(self.parameters['num_bars'])
        self.generate_button = self.findChild(QtWidgets.QPushButton, 'generate_pushbutton')
        self.scale_combobox = self.findChild(QtWidgets.QComboBox, 'scale_combobox')
        self.scale_combobox.setCurrentText(self.parameters['scale'])
        self.generate_error_message = self.findChild(QtWidgets.QLabel, 'generate_error_message')
        self.title_parameter = self.findChild(QtWidgets.QPlainTextEdit, 'titleparameter_textedit')
        self.dissonace_slider = self.findChild(QtWidgets.QSlider, 'dissonance_slider')
        self.dissonace_slider.setMaximum(100)
        self.dissonace_slider.setValue(self.parameters['dissonance_value'])
        self.note_density_slider = self.findChild(QtWidgets.QSlider, 'note_density_slider')
        self.note_density_slider.setMaximum(100)
        self.note_density_slider.setValue(self.parameters['note_density'])
        self.preview_play_button = self.findChild(QtWidgets.QPushButton, 'preview_play_button')
        self.generated_piece_preview = self.findChild(QtWidgets.QLabel, 'generated_piece_preview')
        self.save_to_library_button = self.findChild(QtWidgets.QPushButton, 'save_to_library_button')
        self.preview_time_label = self.findChild(QtWidgets.QLabel, 'preview_time_label')
        self.library_full_label = self.findChild(QtWidgets.QLabel, 'library_full_label')
        self.preview_piece_title = self.findChild(QtWidgets.QLabel, 'preview_piece_title')
        self.preview_prev_pg_button = self.findChild(QtWidgets.QPushButton, 'preview_prev_pg')
        self.preview_prev_pg_button.setEnabled(False)
        self.preview_next_pg_button = self.findChild(QtWidgets.QPushButton, 'preview_next_pg')
        self.preview_next_pg_button.setEnabled(False)
        self.preview_pg_num_label = self.findChild(QtWidgets.QLabel, 'preview_pg_num_label')
        self.preview_pg_num_label.setText('Page 0/0')
        self.tab_widget = self.findChild(QtWidgets.QTabWidget, 'Tabs')
        self.disabled_buttons_label = self.findChild(QtWidgets.QLabel, 'disabled_buttons_label')
        self.transformer_disable_label = self.findChild(QtWidgets.QLabel, 'transformer_disable_label')
        self.model_combobox = self.findChild(QtWidgets.QComboBox, 'model_combobox')
        self.model_combobox.setCurrentText(self.parameters['model'])
        # connect widgets to functions
        self.generate_info_button.clicked.connect(self.open_user_manual)
        self.numbars_spinbox.valueChanged.connect(self.update_num_bars_parameter)
        self.generate_button.clicked.connect(self.generate_bars)
        self.dissonace_slider.sliderMoved.connect(self.dissonance_slider_moved)
        self.note_density_slider.sliderMoved.connect(self.note_density_slider_moved)
        self.preview_play_button.clicked.connect(self.preview_play)
        self.save_to_library_button.clicked.connect(self.save_piece)
        self.scale_combobox.currentIndexChanged.connect(self.scale_combobox_changed)
        self.model_combobox.currentIndexChanged.connect(self.model_select)
        self.title_parameter.textChanged.connect(self.validate_title_parameter)
        self.preview_prev_pg_button.clicked.connect(self.preview_prev_page)
        self.preview_next_pg_button.clicked.connect(self.preview_next_page)
        # initialising processes
        self.tab_widget.currentChanged.connect(self.tab_changed)
        self.disable_preview_buttons()
        self.set_transformer_mask_label()
        # Library Tab----------------------------------------------------------------------------------------------
        # define and initialise Library Tab widgets
        self.library_info_button = self.findChild(QtWidgets.QPushButton, 'library_info_button')
        self.current_piece = self.findChild(QtWidgets.QLabel, 'current_piece_textlabel')
        self.current_piece.hide()
        self.play_button = self.findChild(QtWidgets.QPushButton, 'play_button')
        self.play_button.setEnabled(False)
        self.open_score_button = self.findChild(QtWidgets.QPushButton, 'open_score_button')
        self.open_score_button.setEnabled(False)
        self.delete_button = self.findChild(QtWidgets.QPushButton, 'delete_button_pushbutton')
        self.delete_button.setEnabled(False)
        self.slots_display = self.findChild(QtWidgets.QLabel, 'slots_occupied_display')
        self.library_pdf_label = self.findChild(QtWidgets.QLabel, 'library_pdf_label')
        self.library_prev_pg_button = self.findChild(QtWidgets.QPushButton, 'library_prev_pg')
        self.library_next_pg_button = self.findChild(QtWidgets.QPushButton, 'library_next_pg')
        self.library_pg_num_label = self.findChild(QtWidgets.QLabel, 'library_pg_num_label')
        self.time_remaining_label = self.findChild(QtWidgets.QLabel, 'time_remaining_label')
        # connect widgets to functions
        self.library_info_button.clicked.connect(self.open_user_manual)
        self.play_button.clicked.connect(self.play_midi)
        self.open_score_button.clicked.connect(self.open_score)
        self.delete_button.clicked.connect(self.delete_piece)
        self.library_prev_pg_button.clicked.connect(self.library_prev_page)
        self.library_next_pg_button.clicked.connect(self.library_next_page)
        # initialising processes
        self.define_slots()
        self.num_slots_occupied = len(os.listdir('MIDI Output'))
        self.current_selected = None  # variable to keep track of which piece is currently selected
        self.current_piece.setText(self.current_selected)
        self.midi_path = None
        self.abj_path = None
        self.reset_slots()
        self.library_next_pg_button.setEnabled(False)
        self.library_prev_pg_button.setEnabled(False)
        self.load_blank_library_pdf()
        self.set_title_parameter()
        self.check_library_count()
        self.show()
        pygame.init()

    # Generate Tab methods-------------------------------------------------------------------------------

    # Next page button
    def preview_next_page(self):
        self.selected_preview_page += 1
        self.load_preview_pdf_image()
        self.preview_update_page_buttons()
        self.preview_pg_num_label.setText(f'Page {self.selected_preview_page + 1}/{self.temp_pdf_pages}')

    # Previous page button
    def preview_prev_page(self):
        self.selected_preview_page -= 1
        self.load_preview_pdf_image()
        self.preview_update_page_buttons()
        self.preview_pg_num_label.setText(f'Page {self.selected_preview_page + 1}/{self.temp_pdf_pages}')

    # Enables/diables buttons if on first or last page
    def preview_update_page_buttons(self):
        # disable next page button if on last page
        if self.selected_preview_page + 1 >= self.temp_pdf_pages:
            self.preview_next_pg_button.setEnabled(False)
            next_enabled_flag = False
        else:
            self.preview_next_pg_button.setEnabled(True)
            next_enabled_flag = True
        # disable prev page button if on first page
        if self.selected_preview_page <= 0:
            self.preview_prev_pg_button.setEnabled(False)
            prev_enabled_flag = False
        else:
            self.preview_prev_pg_button.setEnabled(True)
            prev_enabled_flag = True

    # hides or reveals note dissonance and density depending on selected model
    def set_transformer_mask_label(self):
        if self.model_combobox.currentText() == 'GAN':
            self.transformer_disable_label.hide()
        elif self.model_combobox.currentText() == 'Transformer':
            self.transformer_disable_label.show()
    # Load PDF score
    def load_preview_pdf_image(self):
        # get number of pages in PDF
        # Open the PDF file in read-binary mode
        with open(self.temp_abjad_path, 'rb') as pdf_file:
            # Create a PdfFileReader object to read the file
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            # Get the number of pages in the file
            self.temp_pdf_pages = len(pdf_reader.pages)
        doc = fitz.open(self.temp_abjad_path)
        pixmap = doc.load_page(self.selected_preview_page).get_pixmap()
        # Convert PyMuPDF pixmap to PyQt pixmap
        qimage = QtGui.QImage(pixmap.samples, pixmap.width, pixmap.height, pixmap.stride, QtGui.QImage.Format.Format_RGB888)
        qpixmap = QtGui.QPixmap.fromImage(qimage)
        self.generated_piece_preview.setPixmap(qpixmap)

    # disable play & save buttons
    def disable_preview_buttons(self):
        self.save_to_library_button.setEnabled(False)
        self.preview_play_button.setEnabled(False)
        self.preview_next_pg_button.setEnabled(False)
        self.preview_prev_pg_button.setEnabled(False)
        self.disabled_buttons_label.show()
        self.generated_piece_preview.hide()

    # Set total time label
    def preview_set_time_label(self):
        # set total duration label
        mid = mido.MidiFile(self.temp_midi_path)
        total_length = int(mid.length)
        self.preview_time_label.setText(convert_to_min(total_length))
        self.preview_pg_num_label.setText(f'{self.selected_preview_page + 1}/{self.temp_pdf_pages}')

    # Play midi file
    def preview_play(self):
        pygame.mixer.music.load(self.temp_midi_path)
        pygame.mixer.music.play()
        self.preview_play_button.setText('Stop')
        self.preview_play_button.clicked.disconnect(self.preview_play)
        self.preview_play_button.clicked.connect(self.preview_stop)
        # set a timer for piece when playing
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.preview_time_increment)
        self.timer.start(1000)

    # Stop playing midi file
    def preview_stop(self):
        pygame.mixer.music.stop()
        self.timer.timeout.disconnect(self.preview_time_increment)
        self.preview_play_button.setText('Play')
        self.preview_play_button.clicked.disconnect(self.preview_stop)
        self.preview_play_button.clicked.connect(self.preview_play)
        mid = mido.MidiFile(self.temp_midi_path)
        total_length = int(mid.length)
        self.preview_time_label.setText(convert_to_min(total_length))

    # Timer countdown when midi is playing
    def preview_time_increment(self):
        if pygame.mixer.music.get_busy():
            mid = mido.MidiFile(self.temp_midi_path)
            total_length = int(mid.length)
            current_second = int(pygame.mixer.music.get_pos() / 1000)
            self.preview_time_label.setText(convert_to_min(total_length - current_second))
        else:
            self.preview_play_button.clicked.disconnect(self.preview_stop)
            self.preview_play_button.clicked.connect(self.preview_play)
            self.preview_play_button.setText('Play')
            mid = mido.MidiFile(self.temp_midi_path)
            total_length = int(mid.length)
            self.preview_time_label.setText(convert_to_min(total_length))
            self.timer.timeout.disconnect(self.preview_time_increment)

    # Save piece to library and remove from Generate tab preview
    def save_piece(self):
        if pygame.mixer.music.get_busy():
            self.preview_stop()
        shutil.move(self.temp_midi_path, 'Midi Output')
        shutil.move(self.temp_abjad_path, 'Abjad Output')
        self.reset_slots()
        self.set_title_parameter()
        self.disable_preview_buttons()
        self.preview_time_label.setText('00:00')
        self.preview_piece_title.setVisible(False)
        self.preview_pg_num_label.setText('Page 0/0')
        self.check_library_count()
        self.disabled_buttons_label.show()

    # Delete PDF and midi files from temporary output directories
    def remove_temp_files(self):
        for path in os.listdir('Data/Temporary Abjad Output'):
            os.remove('Data/Temporary Abjad Output/' + path)
        for path in os.listdir('Data/Temporary Midi Output'):
            os.remove('Data/Temporary Midi Output/' + path)

    # Calls music generation module to generate new piece
    def generate_bars(self):
        self.generate_button.setEnabled(False)
        self.remove_temp_files()
        # determine if the chosen scale uses sharps or flats
        self.use_sharp_dict = False
        if self.scale_combobox.currentText() in ["C Major", "G Major", "D Major", "A Major", "E Major", "B Major", "A Minor",
            "E Minor", "B Minor", "F# Minor", "C# Minor", "G# Minor"]:
            self.use_sharp_dict = True
        # make sure piece title has not been used already
        if self.title_parameter.toPlainText() not in self.title_array:
            self.generate_button.setEnabled(False)
            if self.model_combobox.currentText() == 'GAN':
                # use GAN to generate new piece
                music_sections = mg.generate_music(self.numbars_spinbox.value(), # number of bars
                                                   self.scale_combobox.currentText(), # scale
                                                   self.dissonace_slider.value(), # dissonance value
                                                   self.note_density_slider.value(),
                                                   ) # note_density
                # turn GAN images into midi & PDF files
                new_abj_path = ap.assemble_score(music_sections,
                                                 self.title_parameter.toPlainText(),
                                                 self.use_sharp_dict,
                                                 self.parameters['scale'])
                new_midi_path = am.assemble_midi(music_sections ,
                                                 self.title_parameter.toPlainText(),
                                                 model='GAN')
            elif self.model_combobox.currentText() == 'Transformer':
                # use Transformer to generate new piece
                new_midi_path, new_abj_path = ts.transformer_generate(self.numbars_spinbox.value(),
                                                                      self.title_parameter.toPlainText(),
                                                                      self.scale_combobox.currentText())
            # temporary paths is [midi filepath, PDF filepath]
            # reset GUI elements to match new piece
            self.temp_midi_path = new_midi_path
            self.temp_abjad_path = new_abj_path
            self.selected_preview_page = 0
            self.generate_button.setEnabled(True)
            self.load_preview_pdf_image()
            self.preview_set_time_label()
            self.save_to_library_button.setEnabled(True)
            self.preview_play_button.setEnabled(True)
            self.preview_piece_title.setText(self.title_parameter.toPlainText())
            self.preview_update_page_buttons()
            self.preview_pg_num_label.setText(f'Page {self.selected_preview_page + 1}/{self.temp_pdf_pages}')
            self.generated_piece_preview.setVisible(True)
            self.preview_piece_title.setVisible(True)
            self.disabled_buttons_label.hide()
            self.indirect_preview_stop()
            self.generate_button.setEnabled(True)

    # display error messages if piece title is not valid
    def validate_title_parameter(self):
        if self.title_parameter.toPlainText() in self.title_array:  # if title is already used
            self.generate_button.setEnabled(False)
            self.generate_error_message.setText('Title already used!')
            self.generate_error_message.show()
        elif len(self.title_parameter.toPlainText()) > 15:  # if title is too long
            self.generate_button.setEnabled(False)
            self.generate_error_message.setText('Title is too long!')
            self.generate_error_message.show()
        else:
            self.generate_button.setEnabled(True)
            self.generate_error_message.hide()

    # Automatically suggests 'Untitled n' for piece title
    def set_title_parameter(self):
        if len(os.listdir('MIDI Output')) == 1:
            self.title_parameter.setPlainText('Untitled 1')
        for i in range(1, len(os.listdir('MIDI Output')) + 1):
            string = 'Untitled ' + str(i) + '.mid'
            if string not in os.listdir('MIDI Output'):
                self.title_parameter.setPlainText(string[:-4])
                break

    # Num bars spinbox widget
    def update_num_bars_parameter(self):
        with open('Data/Parameters.json', 'w') as f:
            self.parameters['num_bars'] = self.numbars_spinbox.value()
            json.dump(self.parameters, f)

    # Dissonance slider widget
    def dissonance_slider_moved(self):
        with open('Data/Parameters.json', 'w') as f:
            self.parameters['dissonance_value'] = self.dissonace_slider.value()
            json.dump(self.parameters, f)

    # Model selection parameter
    def model_select(self):
        with open('Data/Parameters.json', 'w') as f:
            self.parameters['model'] = self.model_combobox.currentText()
            json.dump(self.parameters, f)

        if self.model_combobox.currentText() == 'GAN':
        # enable note density and dissonance
            self.note_density_slider.setEnabled(True)
            self.dissonace_slider.setEnabled(True)
            self.transformer_disable_label.hide()
        elif self.model_combobox.currentText() == 'Transformer':
        # disable note density and dissonance
            self.note_density_slider.setEnabled(False)
            self.note_density_slider.setEnabled(True)
            self.transformer_disable_label.show()

    # note densiity slider widget
    def note_density_slider_moved(self):
        with open('Data/Parameters.json', 'w') as f:
            self.parameters['note_density'] = self.note_density_slider.value()
            json.dump(self.parameters, f)

    # Scale combobox widget
    def scale_combobox_changed(self):
        with open('Data/Parameters.json', 'w') as f:
            self.parameters['scale'] = self.scale_combobox.currentText()
            json.dump(self.parameters, f)

    # Disable generate button if number of pieces in library reaches 40
    def check_library_count(self):
        if len(self.title_array) >= 40:
            self.generate_button.setEnabled(False)
            self.save_to_library_button.setEnabled(False)
            self.library_full_label.show()
        else:
            self.generate_button.setEnabled(True)
            self.library_full_label.hide()

    # Stop midi playback if (some) other widgets other than stop button are pressed
    def indirect_preview_stop(self):
        if pygame.mixer.music.get_busy():
            self.preview_stop()

    # Stop midi playback when changing tabs
    def tab_changed(self, index):
        if index == 0:
            self.indirect_preview_stop()
        elif index == 1:
            self.indirect_library_stop()

    # Library Tab methods----------------------------------------------------------------------------------

    # Next page button
    def library_next_page(self):
        self.selected_library_page += 1
        self.load_library_pdf_image()
        self.library_update_page_buttons()
        self.library_pg_num_label.setText(f'Page {self.selected_library_page + 1}/{self.library_pdf_pages}')

    # Previous page button
    def library_prev_page(self):
        self.selected_library_page -= 1
        self.load_library_pdf_image()
        self.library_update_page_buttons()
        self.library_pg_num_label.setText(f'Page {self.selected_library_page + 1}/{self.library_pdf_pages}')

    # Enables/diables buttons if on first or last page
    def library_update_page_buttons(self):
        # disable next page button if on last page
        if self.selected_library_page + 1 >= self.library_pdf_pages:
            self.library_next_pg_button.setEnabled(False)
        else:
            self.library_next_pg_button.setEnabled(True)
        # disable prev page button if on first page
        if self.selected_library_page <= 0:
            self.library_prev_pg_button.setEnabled(False)
        else:
            self.library_prev_pg_button.setEnabled(True)

    # Load blank PDF score
    def load_blank_library_pdf(self):
        doc = fitz.open('Data/Empty Score.pdf')
        pixmap = doc.load_page(0).get_pixmap()
        # Convert PyMuPDF pixmap to PyQt pixmap
        qimage = QtGui.QImage(pixmap.samples, pixmap.width, pixmap.height, pixmap.stride,
                              QtGui.QImage.Format.Format_RGB888)
        qpixmap = QtGui.QPixmap.fromImage(qimage)
        self.library_pdf_label.setPixmap(qpixmap)

    # Load PDF score
    def load_library_pdf_image(self):
        # get number of pages in PDF
        # Open the PDF file in read-binary mode
        with open(self.abj_path, 'rb') as pdf_file:
            # Create a PdfFileReader object to read the file
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            # Get the number of pages in the file
            self.library_pdf_pages = len(pdf_reader.pages)
        doc = fitz.open(self.abj_path)
        pixmap = doc.load_page(self.selected_library_page).get_pixmap()
        # Convert PyMuPDF pixmap to PyQt pixmap
        qimage = QtGui.QImage(pixmap.samples, pixmap.width, pixmap.height, pixmap.stride,
                                QtGui.QImage.Format.Format_RGB888)
        qpixmap = QtGui.QPixmap.fromImage(qimage)
        self.library_pdf_label.setPixmap(qpixmap)

    # Set total time label
    def library_set_time_label(self):
        # set total duration label
        mid = mido.MidiFile(self.midi_path)
        total_length = int(mid.length)
        self.time_remaining_label.setText(convert_to_min(total_length))
        self.library_pg_num_label.setText(f'{self.selected_library_page + 1}/{self.library_pdf_pages}')

    # Load music generation parameters from json file
    def initialise_parameters(self):
        with open('Data/Parameters.json', 'r') as f:
            self.parameters = json.load(f)

    # Define 40 slots to store generated pieces in
    def define_slots(self):
        self.slot_array = []
        for i in range(1, 41):
            # create the button and add it to the array
            slot = self.findChild(QtWidgets.QPushButton, f'slot_{i}')
            self.slot_array.append(slot)
            # connect the button to it function (one function is made for each button)
            slot.clicked.connect(self.make_slot_function(i))
            # customise button appearance
            slot.setStyleSheet("""
            QPushButton {
            font: 700 18pt "Helvetica Neue";
            color: white;
            background-color: rgba(101, 101, 101, 191);
            border-radius: 5;
            }
            QPushButton:hover {
            color: white;
            background-color: rgba(117, 117, 117, 191);
            border-radius: 5;
            }
            QPushButton:pressed {
            color: white;
            background-color: rgba(130, 130, 130, 191);
            border-radius: 5;
            }
            """)
        for slot in self.slot_array:
            slot.hide()

    # Make a function for each of the 40 slots
    def make_slot_function(self, i):
        def slot_function():
            self.select(i)
        return slot_function

    # This function is attached to each slot widget
    def select(self, button_num):
        self.indirect_library_stop()
        index = button_num - 1
        self.current_selected = self.slot_array[index].text()
        self.current_piece.setText(self.current_selected)
        self.set_path()
        pygame.mixer.music.load(self.midi_path)
        self.selected_library_page = 0
        self.load_library_pdf_image()
        self.library_set_time_label()
        self.library_pg_num_label.setText(f'Page {self.selected_library_page + 1}/{self.library_pdf_pages}')
        self.play_button.setEnabled(True)
        self.open_score_button.setEnabled(True)
        self.delete_button.setEnabled(True)
        self.library_update_page_buttons()
        self.library_pg_num_label.show()

    # Open pdf score
    def open_score(self):
        subprocess.Popen(['open', self.abj_path])

    # Open user manual
    def open_user_manual(self):
        subprocess.Popen(['open', 'Data/User Manual.pdf'])

    # Play MIDI file
    def play_midi(self):
        pygame.mixer.music.load(self.midi_path)
        pygame.mixer.music.play()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.library_time_increment)
        self.timer.start(1000)
        self.play_button.setText('Stop')
        self.play_button.clicked.disconnect(self.play_midi)
        self.play_button.clicked.connect(self.stop_midi)

    # Stop MIDI playback
    def stop_midi(self):
        pygame.mixer.music.stop()
        self.timer.timeout.disconnect(self.library_time_increment)
        self.play_button.setText('Play')
        self.play_button.clicked.disconnect(self.stop_midi)
        self.play_button.clicked.connect(self.play_midi)
        mid = mido.MidiFile(self.midi_path)
        total_length = int(mid.length)
        self.time_remaining_label.setText(convert_to_min(total_length))

    # Timer countdown when midi is playing
    def library_time_increment(self):
        if pygame.mixer.music.get_busy() == True:
            mid = mido.MidiFile(self.midi_path)
            total_length = int(mid.length)
            current_second = int(pygame.mixer.music.get_pos() / 1000)
            self.time_remaining_label.setText(convert_to_min(total_length - current_second))
        else:
            self.play_button.setText('Play')
            self.play_button.clicked.disconnect(self.stop_midi)
            self.play_button.clicked.connect(self.play_midi)
            mid = mido.MidiFile(self.midi_path)
            total_length = int(mid.length)
            self.time_remaining_label.setText(convert_to_min(total_length))
            self.timer.timeout.disconnect(self.library_time_increment)

    # Deletes PDF and MIDI of selected piece & other processes to prevent crash
    def delete_piece(self):
        self.indirect_library_stop()
        self.delete_button.setEnabled(False)
        os.remove(self.midi_path)
        os.remove(self.abj_path)
        self.reset_slots()
        self.slot_array[len(os.listdir('MIDI Output')) - 1].hide()  # hide deleted piece button
        self.current_piece.setText('No Piece Selected')
        self.set_title_parameter()
        self.play_button.setEnabled(False)
        self.open_score_button.setEnabled(False)
        self.library_next_pg_button.setEnabled(False)
        self.load_blank_library_pdf()
        self.check_library_count()
        self.time_remaining_label.setText('00:00')
        self.current_piece.setText('')
        self.library_pg_num_label.setText('Page 0/0')

    # Loads in the slots buttons for all generated pieces
    def reset_slots(self):
        self.title_array = []
        for path in os.listdir('MIDI Output'):
            if path != '.DS_Store':
                self.title_array.append(path[:-4])
        self.title_array.sort()
        self.slots_display.setText(str(len(self.title_array)) + '/40')
        self.generate_button.setEnabled(True)
        if len(self.title_array) >= 40:
            self.generate_button.setEnabled(False)
        # show slots
        for i in range(len(self.title_array)):
            if i < 40:
                self.slot_array[i].setText(self.title_array[i])
                self.slot_array[i].show()

    # Sets midi and pdf filepaths to match selected piece
    def set_path(self):
        self.current_piece.setText(self.current_selected)
        self.midi_path = 'MIDI Output/' + self.current_selected + '.mid'
        self.abj_path = 'Abjad Output/' + self.current_selected + '.pdf'

    # Stop midi playback if (some) other widgets other than stop button are pressed
    def indirect_library_stop(self):
        if pygame.mixer.music.get_busy():
            self.stop_midi()

# function for converting duration from seconds to min:sec
def convert_to_min(sec):
    min_digit = str(int(sec/60))
    sec_digit = str(sec%60)
    if len(min_digit) == 1:
        min_string = '0' + min_digit
    else:
        min_string = min_digit
    if len(sec_digit) == 1:
        sec_string = '0' + sec_digit
    else:
        sec_string = sec_digit
    minutes = min_string + ':' + sec_string
    return minutes
