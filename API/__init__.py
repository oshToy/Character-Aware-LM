import tensorflow as tf
from API.init_model import run as init_model
from API.embedding import get_embedding
from API.sentence_embedding import get_sentence_embedding
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

    def words_embedding(self, sentence: str):
        """
        :param sentence: string of words. for example: "Hello World"
        :return: dict of words as keys and value embedding as value.
            word embedding size is (1,1100)
        for example: {
        "Hello":[[..,..]],
        "World":[[..,..]],
        }
        """
        return get_embedding(sentence, self.session, self.m, self.fasttext_model, self.max_word_length, self.char_vocab,
                             self.rnn_state)

    def sentence_embedding(self, sentence):
        # rnn outputs
        return get_sentence_embedding(sentence, self.session, self.m, self.fasttext_model, self.max_word_length,
                                      self.char_vocab,
                                      self.rnn_state)

    def sentence_probability(self, sentence):
        return 'sentence_probability'

    def generate_words(self, num_samples, temperature):
        return generate(num_samples, self.session, self.m, self.fasttext_model, self.max_word_length, self.char_vocab,
                        self.rnn_state,
                        temperature, self.word_vocab)


if __name__ == '__main__':
    model = CharacterAwareLM()
    model.generate_words(100, 0.5)
    model.words_embedding('test')
    model.words_embedding('i love you')
    model.sentence_embedding('i love you')
