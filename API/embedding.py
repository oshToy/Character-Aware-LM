import numpy as np


def sentence_pre_process(sentence):
    return str(sentence).lower()


def get_embedding(sentence, session, m, fasttext_model, max_word_length, char_vocab, rnn_state):
    sentence = sentence_pre_process(sentence)
    merged_embedding_dict = {}
    for word in sentence.split(' '):
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

        merged_embedding = session.run([m.merged_embedding],
                                                    {m.input: char_input,
                                                     m.input2: input_2,
                                                     m.initial_rnn_state: rnn_state})
        merged_embedding_dict[word] = merged_embedding

    return merged_embedding_dict
