import tensorflow as tf
from API.init_model import run as init_model
from API.embedding import get_embedding
from API.sentence_embedding import get_sentence_embedding
from API.sentence_probability import get_sentence_probability
from API.generate import generate
from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec


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


class CharacterAwareLM:

    def __init__(self):
        self.session = None
        self.m = None
        self.fasttext_model = None
        self.max_word_length = None
        self.rnn_state = None
        self.char_vocab = None
        self.word_vocab = None
        self.load_model()

    def load_model(self):
        self.session, self.m, self.fasttext_model, self.max_word_length, self.char_vocab, self.word_vocab = init_model()
        self.rnn_state = self.session.run(self.m.initial_rnn_state)

    def words_embedding(self, words: str) -> dict:
        """
        returns word embedding for multiply words
        :param words:(str)words that separated by space
        :return:(dict of str: tuple) each key is word and each value is embedding tuple of shape (1, 1100)
        Examples:
            model = CharacterAwareLM()
            model.words_embedding(words='i love you')
            returns:
                <class 'dict'>: {
                     'i': [array([[-0.46512628,  0.00265863,  0.0470854 , ..., -0.8260134]], dtype=float32)]
                     'love': [array([[-0.3157598 ,  0.00265863,  0.06454923, ..., -0.78772575]], dtype=float32)]
                     'you': [array([[-0.28884834,  0.00265863,  0.0470854 , ...,  0.98057705]], dtype=float32)]
                     }

        """
        return get_embedding(words, self.session, self.m, self.fasttext_model, self.max_word_length, self.char_vocab,
                             self.rnn_state)

    def sentence_embedding(self, sentence: str):
        """
        returns sentence embedding for single sentence

        :param sentence:(str) single sentence
        :return: <class 'tuple'>: (1, 650) of doubles

        Examples:
            model = CharacterAwareLM()
            model.sentence_embedding(sentence='i love you')
            returns:
                [[-1.12978451e-01  4.30826768e-02  5.30631952e-02  7.80072063e-03
                  -5.06806336e-02  6.68766201e-02  2.98491260e-03  6.28402084e-02
                  -1.58062458e-01  1.45916179e-01  2.89954692e-01 -3.18211094e-02
                   2.82778349e-02 -2.56609358e-02  9.76490006e-02  3.32620 ]]
        """
        return get_sentence_embedding(sentence, self.session, self.m, self.fasttext_model, self.max_word_length,
                                      self.char_vocab,
                                      self.rnn_state)[0][0]

    def sentence_probability(self, sentence: str):
        """
        return sum of logs probabilities of given sentence.
        prob = log(W1|W0) + log(W2|W0, W1) + ... + log(EOS|Wn, Wn-1, wn-2 ,...)
        :param sentence: (str) single sentence
        :return:

        Examples:
            model = CharacterAwareLM()
            model.sentence_probability(sentence='i love you')
            returns:

                        """
        return get_sentence_probability(sentence, self.session, self.m, self.fasttext_model, self.max_word_length,
                                        self.char_vocab,
                                        self.rnn_state, self.word_vocab)

    def generate_sentences(self, num_samples: int, temperature: float) -> []:
        """
        generate sentences by considering temperature
        :param num_samples: (int) number of words to generate
        :param temperature:(float 0.2 => INF. )
        Lower temperatures make the model increasingly confident in its top choices,
         while temperatures greater than 1 decrease confidence.
         0 temperature is equivalent to argmax /max likelihood,
         while infinite temperature corresponds to a uniform sampling.
        :return:(arr of str) each sentence per arr element

        Examples:
            model = CharacterAwareLM()
            model.generate_sentences(num_samples=10, temperature=1)
            returns:
                0 = {str} ' definitive'
                1 = {str} ' like a old panel where the bits sitting'
        """
        return generate(num_samples, self.session, self.m, self.fasttext_model, self.max_word_length, self.char_vocab,
                        self.rnn_state,
                        temperature, self.word_vocab)

    def generate_sentence_with_prefix(self, prefix: str, temperature: float) -> str:
        """
        generate sentences by considering temperature with given start of the sentence
        :param prefix:(str) prefix of sentence, can be sub-sentence as-well
        :param temperature:(float 0.2 => INF. )
        Lower temperatures make the model increasingly confident in its top choices,
         while temperatures greater than 1 decrease confidence.
         0 temperature is equivalent to argmax /max likelihood,
         while infinite temperature corresponds to a uniform sampling.
        :return:(str) single sentence

        Examples:
            model = CharacterAwareLM()
            model.generate_sentence_with_prefix(prefix='i love', temperature=0.2)
            returns:
               'i love lax'
        """
        num_samples = 300  # MAX
        prefix_rnn_state = \
            get_sentence_embedding(prefix, self.session, self.m, self.fasttext_model, self.max_word_length,
                                   self.char_vocab,
                                   self.rnn_state)[1]
        single_sentence = generate(num_samples, self.session, self.m, self.fasttext_model, self.max_word_length,
                                   self.char_vocab,
                                   prefix_rnn_state,
                                   temperature, self.word_vocab, single_sentence=True)[0]
        return prefix + single_sentence


if __name__ == '__main__':
    model = CharacterAwareLM()
    model.generate_sentences(num_samples=100, temperature=0.2)
    model.words_embedding(words='test')
    model.words_embedding(words='i love you')
    model.sentence_embedding(sentence='i love you')
    model.sentence_probability(sentence='i love you')
    model.generate_sentence_with_prefix(prefix='i love', temperature=0.2)


