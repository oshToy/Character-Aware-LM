import numpy as np


def sentence_pre_process(sentence):
    return str(sentence).lower()


def get_sentence_embedding(sentence, session, m, fasttext_model, max_word_length, char_vocab, rnn_state):
    sentence = sentence_pre_process(sentence)
    for word in sentence.split(' '):
        words_tf = fasttext_model.wv[word]
        input_2 = np.reshape((np.concatenate((words_tf, np.zeros(4))).T), (-1, 1))

        char_input = np.zeros((1, 1, max_word_length))
        for i, c in enumerate('{' + word + '}'):
            char_input[0, 0, i] = char_vocab[c]

        rnn_outputs = session.run([m.rnn_outputs],
                                                    {m.input: char_input,
                                                     m.input2: input_2,
                                                     m.initial_rnn_state: rnn_state})
        rnn_outputs

    return rnn_outputs
