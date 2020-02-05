from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf

import model
from data_reader import load_data, DataReader, FasttextModel, DataReaderFastText
import pandas as pd

flags = tf.flags
FLAGS = flags.FLAGS


def define_flags():
    # data
    flags.DEFINE_string('load_model_for_training', None,
                        '(optional) filename of the model to load. Useful for re-starting training from a checkpoint')
    # model params
    flags.DEFINE_integer('rnn_size', 650, 'size of LSTM internal state')
    flags.DEFINE_integer('highway_layers', 2, 'number of highway layers')
    flags.DEFINE_integer('char_embed_size', 15, 'dimensionality of character embeddings')
    flags.DEFINE_string('kernels', '[1,2,3,4,5,6,7]', 'CNN kernel widths')
    flags.DEFINE_string('kernel_features', '[50,100,150,200,200,200,200]', 'number of features in the CNN kernel')
    flags.DEFINE_integer('rnn_layers', 2, 'number of layers in the LSTM')
    flags.DEFINE_float('dropout', 0.5, 'dropout. 0 = no dropout')

    # optimization
    flags.DEFINE_float('learning_rate_decay', 0.5, 'learning rate decay')
    flags.DEFINE_float('learning_rate', 1.0, 'starting learning rate')
    flags.DEFINE_float('decay_when', 1.0, 'decay if validation perplexity does not improve by more than this much')
    flags.DEFINE_float('param_init', 0.05, 'initialize parameters at')
    flags.DEFINE_integer('num_unroll_steps', 20, 'number of timesteps to unroll for')
    flags.DEFINE_integer('batch_size', 25, 'number of sequences to train on in parallel')
    flags.DEFINE_integer('max_epochs', 20, 'number of full passes through the training data')
    flags.DEFINE_float('max_grad_norm', 5.0, 'normalize gradients at')
    flags.DEFINE_integer('max_word_length', 65, 'maximum word length')

    # bookkeeping
    flags.DEFINE_integer('seed', 3435, 'random number generator seed')
    flags.DEFINE_integer('print_every', 5, 'how often to print current loss')
    flags.DEFINE_string('EOS', None,
                        '<EOS> symbol. should be a single unused character (like +) for PTB and blank for others')


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
        'epoch_number': list(),
        'train_loss': list(),
        'validation_loss': list(),
        "epoch_training_time": list(),
        "model_name": list(),
        "learning_rate": list()
    }


def main(print):
    ''' Trains model from data '''
    if not os.path.exists(FLAGS.train_dir):
        os.mkdir(FLAGS.train_dir)
        print('Created training directory' + FLAGS.train_dir)

    # CSV initialize
    pd.DataFrame(FLAGS.flag_values_dict(), index=range(1)).to_csv(FLAGS.train_dir + '/train_parameters.csv')
    epochs_results = initialize_epoch_data_dict()

    fasttext_model_path = None
    if FLAGS.fasttext_model_path:
        fasttext_model_path = FLAGS.fasttext_model_path

    word_vocab, char_vocab, word_tensors, char_tensors, max_word_length, words_list, wers, acoustics = \
        load_data(FLAGS.data_dir, FLAGS.max_word_length, num_unroll_steps=FLAGS.num_unroll_steps, eos=FLAGS.EOS)

    fasttext_model = None
    if 'fasttext' in FLAGS.embedding:
        fasttext_model = FasttextModel(fasttext_path=fasttext_model_path).get_fasttext_model()

        train_ft_reader = DataReaderFastText(words_list=words_list, batch_size=FLAGS.batch_size,
                                             num_unroll_steps=FLAGS.num_unroll_steps,
                                             model=fasttext_model,
                                             data='train', acoustics=acoustics)

        valid_ft_reader = DataReaderFastText(words_list=words_list, batch_size=FLAGS.batch_size,
                                             num_unroll_steps=FLAGS.num_unroll_steps,
                                             model=fasttext_model,
                                             data='valid', acoustics=acoustics)

    train_reader = DataReader(word_tensors['train'], char_tensors['train'],
                              FLAGS.batch_size, FLAGS.num_unroll_steps, wers['train'], word_vocab, char_vocab)



    valid_reader = DataReader(word_tensors['valid'], char_tensors['valid'],
                              FLAGS.batch_size, FLAGS.num_unroll_steps, wers['train'], word_vocab, char_vocab)


    test_reader = DataReader(word_tensors['test'], char_tensors['test'],
                             FLAGS.batch_size, FLAGS.num_unroll_steps, wers['train'], word_vocab, char_vocab)

    print('initialized all dataset readers')

    with tf.Graph().as_default(), tf.Session() as session:

        # tensorflow seed must be inside graph
        tf.set_random_seed(FLAGS.seed)
        np.random.seed(seed=FLAGS.seed)

        ''' build training graph '''
        initializer = tf.random_uniform_initializer(-FLAGS.param_init, FLAGS.param_init)
        with tf.variable_scope("Model", initializer=initializer):
            train_model = model.inference_graph(
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
                dropout=FLAGS.dropout,
                embedding=FLAGS.embedding,
                fasttext_word_dim=300)
            train_model.update(model.loss_graph(train_model.logits, FLAGS.batch_size))

            # scaling loss by FLAGS.num_unroll_steps effectively scales gradients by the same factor.
            # we need it to reproduce how the original Torch code optimizes. Without this, our gradients will be
            # much smaller (i.e. 35 times smaller) and to get system to learn we'd have to scale learning rate and max_grad_norm appropriately.
            # Thus, scaling gradients so that this trainer is exactly compatible with the original
            train_model.update(model.training_graph(train_model.loss * FLAGS.num_unroll_steps,
                                                    FLAGS.learning_rate, FLAGS.max_grad_norm))

        # create saver before creating more graph nodes, so that we do not save any vars defined below
        saver = tf.train.Saver(max_to_keep=50)

        ''' build graph for validation and testing (shares parameters with the training graph!) '''
        with tf.variable_scope("Model", reuse=True):
            valid_model = model.inference_graph(
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
                dropout=0.0,
                embedding=FLAGS.embedding,
                fasttext_word_dim=300)
            valid_model.update(model.loss_graph(valid_model.logits, FLAGS.batch_size))

        if FLAGS.load_model_for_training:
            saver.restore(session, FLAGS.load_model_for_training)
            string = str('Loaded model from' + str(FLAGS.load_model_for_training) + 'saved at global step' + str(
                train_model.global_step.eval()))
            print(string)
        else:
            tf.global_variables_initializer().run()
            session.run(train_model.clear_char_embedding_padding)
            string = str('Created and initialized fresh model. Size:' + str(model.model_size()))
            print(string)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=session.graph)

        ''' take learning rate from CLI, not from saved graph '''
        session.run(
            tf.assign(train_model.learning_rate, FLAGS.learning_rate),
        )

        ''' training starts here '''
        best_valid_loss = None
        rnn_state = session.run(train_model.initial_rnn_state)
        for epoch in range(FLAGS.max_epochs):

            epoch_start_time = time.time()
            avg_train_loss = 0.0
            count = 0
            for batch_kim, batch_ft in zip(train_reader.iter(), train_ft_reader.iter()):
                x, y = batch_kim
                count += 1
                start_time = time.time()
                if fasttext_model:
                    ft_vectors = fasttext_model.wv[words_list['train'][count]].reshape(fasttext_model.wv.vector_size, 1)
                    loss, _, rnn_state, gradient_norm, step, _ = session.run([
                        train_model.loss,
                        train_model.train_op,
                        train_model.final_rnn_state,
                        train_model.global_norm,
                        train_model.global_step,
                        train_model.clear_char_embedding_padding
                    ], {
                        train_model.input2: batch_ft,
                        train_model.input: x,
                        train_model.targets: y,
                        train_model.initial_rnn_state: rnn_state
                    })
                else:
                    loss, _, rnn_state, gradient_norm, step, _ = session.run([
                        train_model.loss,
                        train_model.train_op,
                        train_model.final_rnn_state,
                        train_model.global_norm,
                        train_model.global_step,
                        train_model.clear_char_embedding_padding
                    ], {
                        train_model.input: x,
                        train_model.targets: y,
                        train_model.initial_rnn_state: rnn_state
                    })

                avg_train_loss += 0.05 * (loss - avg_train_loss)

                time_elapsed = time.time() - start_time

                if count % FLAGS.print_every == 0:
                    string = str(
                        '%6d: %d [%5d/%5d], train_loss = %6.8f secs/batch = %.4fs' % (
                            step,
                            epoch, count,
                            train_reader.length,
                            loss,
                            time_elapsed
                            ))
                    print(string)
            string = str('Epoch training time:' + str(time.time() - epoch_start_time))
            print(string)
            epochs_results['epoch_training_time'].append(str(time.time() - epoch_start_time))

            # epoch done: time to evaluate
            avg_valid_loss = 0.0
            count = 0
            rnn_state = session.run(valid_model.initial_rnn_state)
            for batch_kim, batch_ft in zip(valid_reader.iter(), valid_ft_reader.iter()):
                x, y = batch_kim
                count += 1
                start_time = time.time()

                loss, rnn_state = session.run([
                    valid_model.loss,
                    valid_model.final_rnn_state
                ], {
                    valid_model.input2: batch_ft,
                    valid_model.input: x,
                    valid_model.targets: y,
                    valid_model.initial_rnn_state: rnn_state,
                })

                if count % FLAGS.print_every == 0:
                    string = str("\t> validation loss = %6.8f" % (loss))
                    print(string)
                avg_valid_loss += loss / valid_reader.length

            print("at the end of epoch:" + str(epoch))
            epochs_results['epoch_number'].append(str(epoch))
            print("train loss = %6.8f" % (avg_train_loss))
            epochs_results['train_loss'].append(avg_train_loss)
            print("validation loss = %6.8f" % (avg_valid_loss))
            epochs_results['validation_loss'].append(avg_valid_loss)

            save_as = '%s/epoch%03d_%.4f.model' % (FLAGS.train_dir, epoch, avg_valid_loss)
            saver.save(session, save_as)
            print('Saved model' + str(save_as))
            epochs_results['model_name'].append(str(save_as))
            epochs_results['learning_rate'].append(str(session.run(train_model.learning_rate)))


            ''' write out summary events '''
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="train_loss", simple_value=avg_train_loss),
                tf.Summary.Value(tag="valid_loss", simple_value=avg_valid_loss),
            ])
            summary_writer.add_summary(summary, step)

            ''' decide if need to decay learning rate '''
            if best_valid_loss is not None and np.exp(avg_valid_loss) > np.exp(best_valid_loss) - FLAGS.decay_when:
                print('validation perplexity did not improve enough, decay learning rate')
                current_learning_rate = session.run(train_model.learning_rate)
                string = str('learning rate was:' + str(current_learning_rate))
                print(string)
                current_learning_rate *= FLAGS.learning_rate_decay
                if current_learning_rate < 1.e-6:
                    print('learning rate too small - stopping now')
                    break

                session.run(train_model.learning_rate.assign(current_learning_rate))
                string = str('new learning rate is:' + str(current_learning_rate))
                print(string)
            else:
                best_valid_loss = avg_valid_loss

    # Save model performance data
    pd.DataFrame(epochs_results).to_csv(FLAGS.train_dir + '/train_results.csv')


if __name__ == "__main__":
    tf.app.run()
