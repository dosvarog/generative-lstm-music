#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
#
#  data.py
#
#  Copyright 2017 domagoj <domagoj@domagoj-x1>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#

import music21 as m21
import numpy as np
import pprint

# user can choose what will be interval for splitting music
# supported are 8th's, 16th's and 32nd's
split_intervals = {
        8: 0.5,
        16: 0.25,
        32: 0.125
    }

# index of piano key corresponds to MIDI code shifted by 21
# A4 corresponds to MIDI code 69 and it is 48th key on a piano keyboard
# (counting from 0), so 69 - 21 = 48.
midi_piano_shift = 21
# number of piano keys, this is also dimension of our input vector
num_piano_keys = 88

class Data(object):
    def __init__(self, vector_function=None, split_interval=16):
        self.split_interval = split_interval
        if vector_function is None:
            raise AttributeError("Unknown 'vector_function' attribute! Have you"
                                 " provided (correct) function?")
        self.vector_function = vector_function

    def _read_data(self, filename):
        print("Reading file {0}".format(filename))
        f = m21.converter.parse(filename)
        return f

    def _create_transitional_matrices(self, path):
        def skip_tuplet(c):
            for elem in c.recurse().notesAndRests:
                if elem.duration.tuplets:
                    return True
            return False

        def skip_split_interval(c):
            for elem in c.recurse().notesAndRests:
                if elem.quarterLength < split_intervals[self.split_interval]:
                    return True
            return False

        compositions_by_parts = []
        dataset_size = len(path)
        for current_file, filename in enumerate(path, start=1):
            composition = self._read_data(filename)
            print("{0}/{1}".format(current_file, dataset_size))

            if skip_tuplet(composition):
                print("Skipping file {0} due to tuplets.".format(filename))
                continue
            elif skip_split_interval(composition):
                print("Skipping file {0} due to lower note duration then "
                      "split_interval value.".format(filename))
                continue

            # transpose to C major / a minor
            k = composition.analyze('key')
            if k.mode == 'major':
                interval = m21.interval.Interval(k.tonic, m21.pitch.Pitch('C'))
            elif k.mode == 'minor':
                interval = m21.interval.Interval(k.tonic, m21.pitch.Pitch('a'))
            # maybe do this in place, since there is argument for that?
            composition = composition.transpose(interval)
            nk = composition.analyze('key')
            print("Transposing file {0} from key {1} to key {2}.".format(
                                                               filename, k, nk))
            restructured_composition = composition.voicesToParts()
            parts = restructured_composition.getElementsByClass(m21.stream.Part)
            compositions_by_parts.append(parts)
        
        transitional_tensor = []
        for composition in compositions_by_parts:
            quantized_composition = []
            for part in composition:
                quantized_part = []
                for note in part.recurse().notesAndRests:
                    num_of_quants = int(note.quarterLength /
                                        split_intervals[self.split_interval])
                    q_length_list = [split_intervals[
                                     self.split_interval]] * num_of_quants
                    split = note.splitByQuarterLengths(q_length_list,
                                                       addTies=False)
                    split_notes = []
                    for i, q_note in enumerate(split):
                        if len(split) == 1:
                            split_notes.append((q_note, 'SK'))
                        # maybe we don't even need len(split) > 1 but it works
                        elif i == 0 and len(split) > 1:
                            split_notes.append((q_note, 'S'))
                        elif i == (len(split) - 1):
                            split_notes.append((q_note, 'K'))
                        else:
                            split_notes.append((q_note, ''))
                    quantized_part.extend(split_notes)
                quantized_composition.append(quantized_part)
            transitional_tensor.append(quantized_composition)
        return transitional_tensor

    def _column(self, matrix, col):
        return [row[col] for row in matrix]

    def create_state_matrices(self, path):
        transitional_tensor = self._create_transitional_matrices(path)

        state_tensor = []
        for composition in transitional_tensor:
            piano_roll = []
            for i in range(len(composition[0])):
                quant = self._column(composition, i)
                vector = self.vector_function(quant)
                piano_roll.append(vector)
            state_tensor.append(np.array(piano_roll))
        return state_tensor

    def generate_batches(self, compositions_list, seq_length=64, batch_size=5,
                         validation_size=None, shorten_vectors=True,
                         pad_dataset=True):
        # we create one big composition by concatenating all compositions,
        # which we then split into an array of seq_length long arrays
        print("Generating batches...")
        compositions = np.concatenate(self.create_state_matrices(
                                                             compositions_list))
        length = compositions.shape[0]

        if shorten_vectors:
            zero_columns_idxs = np.where(~compositions.any(axis=0))[0]
            columns_indices = list(range(compositions.shape[1]))
            non_zero_col_idxs = sorted(set(columns_indices).difference(
                                                             zero_columns_idxs))
            first_non_zero_col_idx = non_zero_col_idxs[0]
            last_non_zero_col_idx = non_zero_col_idxs[-1]
            compositions = compositions[:,
                                 first_non_zero_col_idx:last_non_zero_col_idx+1]
            vector_size = compositions.shape[1]
        else:
            vector_size = num_piano_keys


        def create_batches(dataset):
            dataset_length = dataset.shape[0]
            if pad_dataset:
                # we calculate the mod so we can account for padding
                mod = dataset_length % seq_length
                if mod != 0:
                    padding_size = seq_length - mod
                    padding = np.zeros((padding_size, vector_size),
                                       dtype=np.int)
                    dataset = np.concatenate((dataset, padding))
                    # recalculate length
                    dataset_length = dataset.shape[0]
            
            print("This dataset is {0} vectors big".format(dataset_length))

            dataset_y = np.roll(dataset, -1, axis=0)
            dataset_y[-1, :] = 0
            
            dataset_length = dataset.shape[0]

            n = dataset_length // seq_length
            sections = [i * seq_length for i in range(1, n)]
            split_x = np.vsplit(dataset, sections)
            split_y = np.vsplit(dataset_y, sections)
            
            if len(split_x) < batch_size:
                print("Number of items in a batch is smaller than 'batch_size',"
                      " size of batch is batch_size = {0}".format(len(split_x)))
            elif len(split_x) % batch_size != 0:
                print("Last batch will be of size batch_size = {0}".format(
                                                     len(split_x) % batch_size))
            else:
                print("All batches are of the same size")

            batches_x = []
            batches_y = []
            batch_x = []
            batch_y = []
            for i, (sx, sy) in enumerate(zip(split_x, split_y), start=1):
                batch_x.append(sx)
                batch_y.append(sy)
                if (i % batch_size) == 0:
                    batches_x.append(np.array(batch_x))
                    batches_y.append(np.array(batch_y))
                    # mozda dodati del batch zbog sakupljaca smeca?
                    batch_x = []
                    batch_y = []
            # if something is left in batch
            if batch_x:
                batches_x.append(np.array(batch_x))
            if batch_y:
                batches_y.append(np.array(batch_y))

            return batches_x, batches_y

        if validation_size:
            s = int(round((1 - validation_size) * length))
#            s = int((1 - validation_size) * length)
            train_set, validation_set = np.vsplit(compositions, [s])

            train_batches_x, train_batches_y = create_batches(train_set)
            valid_batches_x, valid_batches_y = create_batches(validation_set)

            if shorten_vectors:
                return ((first_non_zero_col_idx, last_non_zero_col_idx),
                        train_batches_x, train_batches_y, valid_batches_x,
                        valid_batches_y)

            return (train_batches_x, train_batches_y, valid_batches_x,
                    valid_batches_y)

        batches_x, batches_y = create_batches(compositions)
        if shorten_vectors:
            return ((first_non_zero_col_idx, last_non_zero_col_idx),
                    batches_x, batches_y)

        return batches_x, batches_y
    
    def piano_roll_to_midi(self, filename, piano_roll):
        single_rolls = {}
        for i in range(num_piano_keys):
            one_roll = self._column(piano_roll, i)
            if any(one_roll):
                single_rolls[i] = one_roll

        streams = [m21.stream.Stream() for _ in range(len(single_rolls))]

        def add_note_rest(duration, rest=False):
            if rest:
                nr = m21.note.Rest(quarterLength=duration)
            else:
                nr = m21.note.Note(quarterLength=duration)
                nr.pitch.midi = key_pitch + midi_piano_shift
            s.append(nr)
        
        for (key_pitch, roll), s in zip(single_rolls.items(), streams):
            previous_token = roll[0]
            duration = 0
            for step, note in enumerate(roll):
                if note == 0:
                    if previous_token == note:
                        duration += split_intervals[self.split_interval]
                        previous_token = note
                    else:
                        add_note_rest(duration)
                        duration = split_intervals[self.split_interval]
                        previous_token = note
                elif note == 1:
                    if previous_token == note:
                        duration += split_intervals[self.split_interval]
                        previous_token = note
                    elif previous_token == 0:
                        add_note_rest(duration, rest=True)
                        duration = split_intervals[self.split_interval]
                        previous_token = note
                    else:
                        add_note_rest(duration)
                        duration = split_intervals[self.split_interval]
                        previous_token = note
                elif note == -1:
                    if previous_token == note:
                        duration += split_intervals[self.split_interval]
                        previous_token = note
                    elif previous_token == 1:
                        add_note_rest(duration)
                        duration = split_intervals[self.split_interval]
                        previous_token = note
                    else:
                        # cannot happen that token 0 comes before -1, but
                        # we process this anyway
                        add_note_rest(duration)
                        duration = split_intervals[self.split_interval]
                        previous_token = note
            # add last note/rest
            if previous_token == 0:
                add_note_rest(duration, rest=True)
            else:
                add_note_rest(duration)
        
        composition = m21.stream.Score()
        for s in streams:
            composition.insert(0, s)
        composition.write('midi', fp=filename)
        print("MIDI file created.")


class InputVector(object):
    # class variables
    last_notes = [0] * num_piano_keys

    @staticmethod
    def bin_vec_integer(quant):
        vector = [0] * num_piano_keys

        for i, note in enumerate(quant):
            if 'Rest' in note[0].classes:
                continue
            for pitch in note[0].pitches:
                piano_key = pitch.midi - midi_piano_shift
                if note[1] == 'S':
                    if (InputVector.last_notes[piano_key] == 0 or
                            InputVector.last_notes[piano_key] == -1):
                        vector[piano_key] = 1
                    elif InputVector.last_notes[piano_key] == 1:
                        vector[piano_key] = -1
                elif note[1] == 'SK':
                    if (InputVector.last_notes[piano_key] == 0 or
                            InputVector.last_notes[piano_key] == -1):
                        vector[piano_key] = 1
                    elif InputVector.last_notes[piano_key] == 1:
                        vector[piano_key] = -1
                else:
                    vector[piano_key] = InputVector.last_notes[piano_key]
        InputVector.last_notes = vector[:]

        return vector

    @staticmethod
    def bin_vec_real(quant):
        vector = [0] * num_piano_keys
        
        for i, note in enumerate(quant):
            if 'Rest' in note[0].classes:
                continue
            for pitch in note[0].pitches:
                piano_key = pitch.midi - midi_piano_shift
                if note[1] == 'S' or note[1] == '':
                    vector[piano_key] = 0.5
                elif note[1] == 'K' or note[1] == 'SK':
                    vector[piano_key] = 1

        return vector


if __name__ == '__main__':
    d = Data(vector_function=InputVector.bin_vec_integer)
    skladbe = ['../dataset/blistaj_akordi.xml', '../dataset/probica.xml']
#    mtrx = d.create_state_matrices(skladbe)
    np.set_printoptions(threshold=np.inf, linewidth=300)
#    d.piano_roll_to_midi('uh.mid', mtrx[0])
    indices, tbx, tby, vbx, vby = d.generate_batches(skladbe, batch_size=5,
                                                     validation_size=0.2,
                                                     pad_dataset=False)
#    print(indices)
#    for a, b in zip(tbx, tby):
#        print("X")
#        print(a.shape)
#        for i in a:
#            print(i.shape)
#        print(str(a).replace('  ', ' '))
#        print()
#        print("--------------------------------------------------------------")
#        print()
#        print("Y")
#        print(b.shape)
#        for i in b:
#            print(i.shape)
#        print(str(b).replace('  ', ' '))
#        print()
#        print("--------------------------------------------------------------")
#        print()
#
#    for a, b in zip(vbx, vby):
#        print("X")
#        print(a.shape)
#        print(str(a).replace('  ', ' '))
#        print()
#        print("--------------------------------------------------------------")
#        print()
#        print("Y")
#        print(a.shape)
#        print(str(b).replace('  ', ' '))
#        print()
#        print("--------------------------------------------------------------")
#        print()

