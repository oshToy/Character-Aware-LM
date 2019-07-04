import logging
import os
import train

ROOT_FOLDER = 'C:\\Users\\Ohad.Volk\\Desktop\\Oshri\\tf-lstm-char-cnn-master\\tf-lstm-char-cnn-master'


def main():
    logger_for_print()
    copy_data_files('PeenTreeBank')# Copy PeenTreeBank corpus to data folder
    train.FLAGS.max_epochs = 50
    train.main('')


def logger_for_print():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler('trainingPennTreeBank.log', 'a'))
    print = logger.info
    return


def copy_data_files(dir_name):
    os.system('robocopy ' + dir_name + ' ' + ROOT_FOLDER + '\\data')
    return

if __name__ == "__main__":
    main()