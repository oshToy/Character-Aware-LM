from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from data_reader import load_data, DataReader, FasttextModel, DataReaderFastText
from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec
import model

flags = tf.flags

# data
flags.DEFINE_string('load_model',
                    r"C:\Users\Ohad.Volk\Desktop\Oshri\tf-lstm-char-cnn-master\tf-lstm-char-cnn-master\trained_models\AMI_2020-03-28--12-43-24\epoch011_3.5723.model",
                    'filename of the model to load')
# we need data only to compute vocabulary
flags.DEFINE_string('data_dir',
                    r"C:\Users\Ohad.Volk\Desktop\Oshri\tf-lstm-char-cnn-master\tf-lstm-char-cnn-master\data_sets\AMI",
                    'data directory')
flags.DEFINE_string('fasttext_model_path',
                    "E:\\RNNLM Project\\Models\\Wikipedia\\Lower Case\\15 Epoch\\Wikipedia_epoch15.model",
                    'fasttext trained model path')
flags.DEFINE_integer('num_samples', 2000, 'how many words to generate')
flags.DEFINE_float('temperature', 1.0, 'sampling temperature')
flags.DEFINE_string('embedding', "kim fasttext", 'embedding method')
# model params
flags.DEFINE_integer('rnn_size', 650, 'size of LSTM internal state')
flags.DEFINE_integer('highway_layers', 2, 'number of highway layers')
flags.DEFINE_integer('char_embed_size', 15, 'dimensionality of character embeddings')
flags.DEFINE_string('kernels', '[1,2,3,4,5,6,7]', 'CNN kernel widths')
flags.DEFINE_string('kernel_features', '[50,100,150,200,200,200,200]', 'number of features in the CNN kernel')
flags.DEFINE_integer('rnn_layers', 2, 'number of layers in the LSTM')
flags.DEFINE_float('dropout', 0.5, 'dropout. 0 = no dropout')
flags.DEFINE_integer('batch_size', 1, 'number of sequences to train on in parallel')
flags.DEFINE_integer('num_unroll_steps', 1, 'number of timesteps to unroll for')

# optimization
flags.DEFINE_integer('max_word_length', 65, 'maximum word length')

# bookkeeping
flags.DEFINE_integer('seed', 3435, 'random number generator seed')
flags.DEFINE_string('EOS', '+',
                    '<EOS> symbol. should be a single unused character (like +) for PTB and blank for others')

FLAGS = flags.FLAGS


class EpochSaver(CallbackAny2Vec):

    def __init__(self, path_prefix):
        self.path_prefix = path_prefix
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        if self.epoch in [1, 5, 10, 15, 20]:
            output_path = get_tmpfile('{}_epoch{}.model'.format(self.path_prefix, self.epoch))
            model.save(output_path)
            print('model save')
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1


def main(_):
    ''' Loads trained model and evaluates it on test split '''

    if FLAGS.load_model is None:
        print('Please specify checkpoint file to load model from')
        return -1

    if not os.path.exists(FLAGS.load_model + '.meta'):
        print('Checkpoint file not found', FLAGS.load_model)
        return -1

    word_vocab, char_vocab, word_tensors, char_tensors, max_word_length, words_list = \
        load_data(FLAGS.data_dir, FLAGS.max_word_length, eos=FLAGS.EOS)

    fasttext_model = FasttextModel(fasttext_path=FLAGS.fasttext_model_path).get_fasttext_model()

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
                dropout=0,
                embedding=FLAGS.embedding,
                fasttext_word_dim=300,
                acoustic_features_dim=4)
            # we need global step only because we want to read it from the model
            global_step = tf.Variable(0, dtype=tf.int32, name='global_step')

        saver = tf.train.Saver()
        saver.restore(session, FLAGS.load_model)
        print('Loaded model from', FLAGS.load_model, 'saved at global step', global_step.eval())

        ''' training starts here '''
        rnn_state = session.run(m.initial_rnn_state)
        logits = np.ones((word_vocab.size,))
        rnn_state = session.run(m.initial_rnn_state)
        for i in range(FLAGS.num_samples):
            logits = logits / FLAGS.temperature
            prob = np.exp(logits)
            prob /= np.sum(prob)
            prob = prob.ravel()
            ix = np.random.choice(range(len(prob)), p=prob)

            word = word_vocab.token(ix)
            words_tf = fasttext_model.wv[word]
            input_2 = np.reshape((np.concatenate((words_tf, np.zeros(4))).T), (-1, 1))
            if word == '|':  # EOS
                print('<unk>', end=' ')
            elif word == '+':
                print('\n')
            else:
                print(word, end=' ')

            char_input = np.zeros((1, 1, max_word_length))
            for i, c in enumerate('{' + word + '}'):
                char_input[0, 0, i] = char_vocab[c]

            logits, rnn_state = session.run([m.logits, m.final_rnn_state],
                                            {m.input: char_input,
                                             m.input2: input_2,
                                             m.initial_rnn_state: rnn_state})
            logits = np.array(logits)


if __name__ == "__main__":
    tf.app.run()
