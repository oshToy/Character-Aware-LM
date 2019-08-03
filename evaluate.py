from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
import pandas as pd
import model
from data_reader import load_data, DataReader

FLAGS = tf.flags.FLAGS


def run_test(session, m, data, batch_size, num_steps):
    """Runs the model on the given data."""

    costs = 0.0
    iters = 0
    state = session.run(m.initial_state)

    for step, (x, y) in enumerate(reader.dataset_iterator(data, batch_size, num_steps)):
        cost, state = session.run([m.cost, m.final_state], {
            m.input_data: x,
            m.targets: y,
            m.initial_state: state
        })

        costs += cost
        iters += 1

    return costs / iters


def initialize_epoch_data_dict():
    return {
        'test_loss': list(),
        'test_perplexity': list(),
        'test_samples': list(),
        'time_elapsed': list(),
        'time_per_batch': list(),
    }

def save_data_to_csv(avg_loss,count,time_elapsed):
    pd.DataFrame(FLAGS.flag_values_dict(), index=range(1)).to_csv(FLAGS.train_dir + '/test_parameters.csv')
    test_results = initialize_epoch_data_dict()

    test_results['test_loss'].append(avg_loss)
    test_results['test_perplexity'].append(np.exp(avg_loss))
    test_results["test_samples"].append(count * FLAGS.batch_size)
    test_results["time_elapsed"].append(time_elapsed)
    test_results["time_per_batch"].append(time_elapsed / count)

    pd.DataFrame(test_results, index=range(1)).to_csv(FLAGS.train_dir + '/test_results.csv')
def main(print, embedding):
    ''' Loads trained model and evaluates it on test split '''
    if FLAGS.load_model_for_test is None:
        print('Please specify checkpoint file to load model from')
        return -1

    if not os.path.exists(FLAGS.load_model_for_test + ".index"):
        print('Checkpoint file not found', FLAGS.load_model_for_test)
        return -1

    word_vocab, char_vocab, word_tensors, char_tensors, max_word_length = \
        load_data(FLAGS.data_dir, FLAGS.max_word_length, eos=FLAGS.EOS)

    test_reader = DataReader(word_tensors['test'], char_tensors['test'],
                              FLAGS.batch_size, FLAGS.num_unroll_steps)

    print('initialized test dataset reader')

    with tf.Graph().as_default(), tf.Session() as session:

        # tensorflow seed must be inside graph
        tf.set_random_seed(FLAGS.seed)
        np.random.seed(seed=FLAGS.seed)

        ''' build inference graph '''
        with tf.variable_scope("Model"):
            m = model.inference_graph(
                    char_vocab_size=char_vocab.size,
                    word_vocab_size=word_vocab.size,
                    char_embed_size=FLAGS.char_embed_size,
                    batch_size=FLAGS.batch_size,
                    num_highway_layers=FLAGS.highway_layers,
                    num_rnn_layers=FLAGS.rnn_layers,
                    rnn_size=FLAGS.rnn_size,
                    max_word_length=max_word_length,
                    kernels=eval(FLAGS.kernels),
                    kernel_features=eval(FLAGS.kernel_features),
                    num_unroll_steps=FLAGS.num_unroll_steps,
                    dropout=0)
            m.update(model.loss_graph(m.logits, FLAGS.batch_size, FLAGS.num_unroll_steps))

            global_step = tf.Variable(0, dtype=tf.int32, name='global_step')

        saver = tf.train.Saver()
        saver.restore(session, FLAGS.load_model_for_test)
        print('Loaded model from' + str(FLAGS.load_model_for_test) + 'saved at global step' +str(global_step.eval()))

        ''' training starts here '''
        rnn_state = session.run(m.initial_rnn_state)
        count = 0
        avg_loss = 0
        start_time = time.time()
        for x, y in test_reader.iter():
            count += 1
            loss, rnn_state = session.run([
                m.loss,
                m.final_rnn_state
            ], {
                m.input  : x,
                m.targets: y,
                m.initial_rnn_state: rnn_state
            })

            avg_loss += loss

        avg_loss /= count
        time_elapsed = time.time() - start_time

        print("test loss = %6.8f, perplexity = %6.8f" % (avg_loss, np.exp(avg_loss)))
        print("test samples:" + str( count*FLAGS.batch_size) + "time elapsed:" + str( time_elapsed) + "time per one batch:" +str(time_elapsed/count))

        save_data_to_csv(avg_loss, count, time_elapsed)



if __name__ == "__main__":
    tf.app.run()
