import numpy as np


def sentence_pre_process(sentence):
    return str(sentence).lower()


def get_sentence_probability(sentence, session, m, fasttext_model, max_word_length, char_vocab,
                             rnn_state, word_vocab, eos=' +'):
    sentence = sentence_pre_process(sentence)
    sentence = sentence + eos
    logits = np.ones((word_vocab.size,))
    prob_list = []
    first_word_flag = True
    for word in sentence.split(' '):

        if not first_word_flag:
            prob = np.exp(logits)
            prob /= np.sum(prob)
            prob = prob.ravel()
            try:
                word_index = word_vocab.__getitem__(word)
            except Exception as e:
                print(e)
                word_index = word_vocab.__getitem__('|')  # UNK

            word_prob = prob[word_index]
            prob_list.append(word_prob)

        first_word_flag = False
        words_tf = fasttext_model.wv[word]
        input_2 = np.reshape((np.concatenate((words_tf, np.zeros(4))).T), (-1, 1))

        if word == '|':
            print('<unk>', end=' ')
        elif word == '+':  # EOS
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

    return prob_list
