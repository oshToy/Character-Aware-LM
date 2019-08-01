import logging
import os
from train import main as train
from evaluate import main as test
import datetime
import json
import tensorflow as tf

CONFIG_FILE = 'config.txt'
flags = tf.flags


def main():
    config_dict = import_config_settings()
    logs_folder = config_dict['logs_folder']
    data_sets_folder = config_dict['data_sets_folder']
    trained_models_folder = config_dict['trained_models_folder']

    for model in config_dict['models']:
        logger = logger_for_print(folder=logs_folder, file_name=config_dict['data_sets_folder'])
        #copy_data_files(data_folder=data_sets_folder + model['data_set'])
        data_set = model['data_set']
        del_all_flags(tf.flags.FLAGS)
        flags.DEFINE_string('data_dir', data_sets_folder + '/' + data_set,
                            'data directory. Should contain train.txt/valid.txt/test.txt with input data')
        flags.DEFINE_string('train_dir', trained_models_folder + '/' + data_set,
                            'training directory (models and summaries are saved there periodically)')
        if 'training' in model and model['training'] == 'True':
            embedding = list(model['embedding'])
            train(logger, embedding)
        if 'testing' in model and model['testing'] == 'True':
            if 'epoch_test' in model:
                test(logger, embedding , model['epoch_test'])
            else:
                test(logger, embedding)


def import_config_settings():
    with open(CONFIG_FILE) as json_file:
        return json.load(json_file)


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


def logger_for_print(folder='', file_name='logger'):
    logging.basicConfig(level=logging.CRITICAL, format='%(message)s')
    logger = logging.getLogger()
    date = str(datetime.datetime.now()).replace(':', '_').replace('.', '_')
    logger.addHandler(logging.FileHandler(folder + '\\' + file_name + date+'.log', 'a'))
    print = logger.critical
    return print


def copy_data_files(data_folder):
    root_folder = os.getcwd()
    os.system('robocopy ' + data_folder + ' ' + root_folder + '/data')
    return


if __name__ == "__main__":
    main()