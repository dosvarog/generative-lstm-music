#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
#
#  lstm_mse2.py
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

import os
import time
import glob
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from data import Data, InputVector, num_piano_keys


DATA_DIR = '../dataset/bach/'
#MODEL_SAVE_DIR = './model_save_mse2/'
MODEL_SAVE_DIR = './jako_trenirani_model_50_epoha/'
LOGS_DIR = './model_logs_mse2/'
MODEL_SAVE_NAME = 'model_mse2.ckpt'

def print_piano_roll(outputs, b_y=None):
    for i, grp in enumerate(outputs):
        np.set_printoptions(linewidth=500)
        l = []
        for step in grp:
            ll = []
            for note in step:
                if note <= -0.5:
                    ll.append(-1)
                elif note > 0.5:
                    ll.append(1)
                else:
                    ll.append(0)
            l.append(ll)

        for x in l:
            print("pred=", str(np.array(x)).replace('   ',' ')
                                           .replace('.','')
                                           .replace('[','')
                                           .replace(']','')
                                           .replace('  ', ' ')
                                           .replace('  ', ' '))
        print("---------------------------------------------------------------")
        if b_y:
            for y in b_y[i]:
                print("real=", str(np.array(y)).replace('   ',' ')
                                               .replace('.','')
                                               .replace('[','')
                                               .replace(']','')
                                               .replace('  ', ' ')
                                               .replace('  ', ' '))
        print("---------------------------------------------------------------")

def read_dataset_files(path=DATA_DIR):
    dataset = glob.glob(path + '*.*')
    return dataset


# batch parameters
BATCH_SIZE = 8


# pickling dataset
PICKLED = True
TRAIN = True
pickle_filename = 'bach_batch_size_' + str(BATCH_SIZE) + '.pickle'

data_ = Data(vector_function=InputVector.bin_vec_integer)
if not PICKLED and TRAIN:
    compositions = read_dataset_files()
    indices, tbx, tby, vbx, vby = data_.generate_batches(compositions,
                                                         batch_size=BATCH_SIZE,
                                                         validation_size=0.2,
                                                         pad_dataset=True)
    # let's pickle this for later
    with open(pickle_filename, 'wb') as f:
        batches = [indices, tbx, tby, vbx, vby]
        pickle.dump(batches, f)

elif TRAIN:
    # we have pickled dataset, let's unpickle it
    with open(pickle_filename, 'rb') as f:
        b = pickle.load(f)
        indices, tbx, tby, vbx, vby = b[0], b[1], b[2], b[3], b[4]

train_size = 1000
in_out_vector_size = len(tbx[0][0][0])

# PARAMETERS
global_step = tf.Variable(0, trainable=False)
INPUT_SIZE = in_out_vector_size
OUTPUT_SIZE = in_out_vector_size
RNN_HIDDEN = 512
learning_rate = 0.001 #tf.train.exponential_decay(0.01, global_step,
                #                           train_size, 0.96, staircase=True)
num_layers = 3


# (batch, time, in)
x = tf.placeholder(tf.float32, (None, None, INPUT_SIZE))
# (batch, time, out)
y = tf.placeholder(tf.float32, (None, None, OUTPUT_SIZE))

def LSTM(x_):
    cell = tf.contrib.rnn.LSTMCell(RNN_HIDDEN, state_is_tuple=True)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8)
    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    batch_size = tf.shape(x_)[0]
    initial_state = cell.zero_state(batch_size, tf.float32)

    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell,
                                                x_,
                                                initial_state=initial_state,
                                                time_major=False)

    final_projection = lambda lx: layers.linear(lx, num_outputs=OUTPUT_SIZE,
                                                activation_fn=tf.nn.tanh)
    predicted_outputs = tf.map_fn(final_projection, rnn_outputs)

    return predicted_outputs


pred = LSTM(x)

# loss and optimiser
loss = tf.reduce_mean(tf.square(pred - y))
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)#,
#                                                        global_step=global_step)

# accuracy
acc = tf.reduce_mean(tf.cast(tf.abs(y - pred) < 0.5, tf.float32))

# tensorboard
tf.summary.scalar('train_loss', loss)
tf.summary.scalar('train_accuracy', acc)
#tf.summary.scalar('valid_loss', loss)
#tf.summary.scalar('valid_acc', acc)
tf.summary.scalar('learning_rate', learning_rate)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

def train(num_epochs=100):
    NUM_OF_EPOCHS = num_epochs

    with tf.Session() as sess:
        # for tensorboard
        summaries = tf.summary.merge_all()
        writer = tf.summary.FileWriter(
                               os.path.join(LOGS_DIR,
                                            time.strftime("%Y-%m-%d-%H-%M-%S")))
        writer.add_graph(sess.graph)
        
        sess.run(init)

        num_batches_train = len(tbx)
        num_batches_valid = len(vbx)
        print("There are {0} batches in train dataset.".format(
                                                             num_batches_train))
        print("There are {0} batches in valid dataset.".format(
                                                             num_batches_valid))
        start_time = time.time()
        for epoch in range(NUM_OF_EPOCHS):
            for i in range(num_batches_train):
                train_batch_x = tbx[i]
                train_batch_y = tby[i]
                sess.run(optimiser, feed_dict={x: train_batch_x,
                                               y: train_batch_y})
                if i % 5 == 0:
                    print("Skup za treniranje:")
                    train_acc, train_loss = sess.run([acc, loss],
                                                feed_dict={x: train_batch_x,
                                                           y: train_batch_y})
#                    train_loss = sess.run(loss, feed_dict={x: train_batch_x,
#                                                           y: train_batch_y})
                    print("Epoha={0:3d}, korak={1}, gubitak={2:10.8f}, "
                          "tocnost={3:10.8f}".format(epoch, i, train_loss,
                                                     train_acc))
    
                # for tensorboard
                summs = sess.run(summaries, feed_dict={x: train_batch_x,
                                                       y: train_batch_y})
                writer.add_summary(summs, epoch * num_batches_train + i)

            for i in range(num_batches_valid):
                valid_batch_x = vbx[i]
                valid_batch_y = vbx[i]
#                sess.run(optimiser, feed_dict={x: valid_batch_x,
#                                               y: valid_batch_y})
                if i % 5 == 0:
                    print("Skup za ispitivanje:")
                    valid_acc, valid_loss = sess.run([acc, loss],
                                                feed_dict={x: valid_batch_x,
                                                           y: valid_batch_y})
#                    valid_loss = sess.run(loss, feed_dict={x: valid_batch_x,
#                                                           y: valid_batch_y})
                    print("Epoha={0:3d}, korak={1}, gubitak={2:10.8f}, "
                          "tocnost={3:10.8f}".format(epoch, i, valid_loss,
                                                     valid_acc))
    
                # for tensorboard
#                summs = sess.run(summaries, feed_dict={x: valid_batch_x,
#                                                       y: valid_batch_y})
#                writer.add_summary(summs, epoch * num_batches_valid + i)

        end_time = time.time()
        print("Finished training in {0} s.".format(end_time - start_time))
        # spremiti model
        save_path = os.path.join(MODEL_SAVE_DIR, MODEL_SAVE_NAME)
        saver.save(sess, save_path)
        print("Model saved to {0}.".format(MODEL_SAVE_DIR))
        
    #    o = sess.run(pred, feed_dict={x: batch_x})
    #    print_piano_roll(o, batch_y)


def generate_music(prime, num=64):
    print("Generating some cool music...")

    # we need to delete this dummy row later 
    generated_song = np.empty((1, 1, in_out_vector_size), dtype=np.int)
    pr = prime

    def clip_composition(composition):
        first_idx = indices[0]
        last_idx = indices[1]
        clipped = composition[0][:, first_idx:last_idx+1]

        return np.array([clipped])

    def quant_to_ints(quant):
        quant_int = []
        for note in quant:
            if note <= -0.5:
                quant_int.append(-1)
            elif note > 0.5:
                quant_int.append(1)
            else:
                quant_int.append(0)

        return np.array([[quant_int]]) 
    
    np.set_printoptions(threshold=np.inf, linewidth=300)
    pr = clip_composition(pr)

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, MODEL_SAVE_DIR + MODEL_SAVE_NAME)
        for seq in range(num):
            o = sess.run(pred, feed_dict={x: pr})
            last = quant_to_ints(o[0][-1])
            generated_song = np.concatenate((generated_song, last), axis=1)
            pr = np.concatenate((pr, last), axis=1)
    print("Music generated.")

    # removing dummy row
    generated_song = np.delete(generated_song, (0), axis=1)

    complete_song = np.concatenate((clip_composition(prime), generated_song),
                                   axis=1)[0]
    print("Complete song", complete_song.shape)

    for_insert = np.zeros((complete_song.shape[0], indices[0]))
    for_append = np.zeros((complete_song.shape[0],
                          num_piano_keys - indices[1] - 1))
    complete_song = np.concatenate((for_insert, complete_song, for_append),
                                    axis=1)
    
    return complete_song

if __name__ == '__main__':
#    train(num_epochs=25)
    pr = ['../dataset/bwv128_5_sjeme.xml']
    gen_data = Data(vector_function=InputVector.bin_vec_integer) 
    prime = np.array(gen_data.create_state_matrices(pr)) 
    new_song = generate_music(prime, num=200)
#    print(new_song)
    data_.piano_roll_to_midi('bacher_bwv128_5.mid', new_song)

