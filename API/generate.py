import numpy as np


def generate(num_samples, session, m, fasttext_model, max_word_length, char_vocab, rnn_state, temperature, word_vocab,
             single_sentence=False):
    logits = np.ones((word_vocab.size,))
    generated_sentences = ['']
    for i in range(num_samples):
        logits = logits / temperature
        prob = np.exp(logits)
        prob /= np.sum(prob)
        prob = prob.ravel()
        ix = np.random.choice(range(len(prob)), p=prob)
        word = word_vocab.token(ix)
        words_tf = fasttext_model.wv[word]
        input_2 = np.reshape((np.concatenate((words_tf, np.zeros(4))).T), (-1, 1))

        if word == '|':  # EOS
            generated_sentences[-1] = generated_sentences[-1] + ' <unk>'
        elif word == '+':
            if single_sentence:
                return generated_sentences
            generated_sentences.append('')
        else:
            generated_sentences[-1] = generated_sentences[-1] + ' ' + word

        char_input = np.zeros((1, 1, max_word_length))
        for i, c in enumerate('{' + word + '}'):
            char_input[0, 0, i] = char_vocab[c]

        logits, rnn_state = session.run([m.logits, m.final_rnn_state],
                                        {m.input: char_input,
                                         m.input2: input_2,
                                         m.initial_rnn_state: rnn_state})
        logits = np.array(logits)

    return generated_sentences
