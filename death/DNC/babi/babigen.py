
import pickle
import numpy as np
from shutil import rmtree
import os
from os import listdir, mkdir
from os.path import join, isfile, isdir, dirname, basename, normpath, abspath, exists
import subprocess
import death.DNC.archi.param as param
from threading import Thread
import time


def create_dictionary(files_list):
    """
    creates a dictionary of unique lexicons in the dataset and their mapping to numbers
    Parameters:
    ----------
    files_list: list
        the list of files to scan through
    Returns: dict
        the constructed dictionary of lexicons
    """

    lexicons_dict = {}
    id_counter = 0

    for indx, filename in enumerate(files_list):
        with open(filename, 'r') as fobj:
            for line in fobj:

                # first seperate . and ? away from words into seperate lexicons
                line = line.replace('.', ' .')
                line = line.replace('?', ' ?')
                line = line.replace(',', ' ')

                for word in line.split():
                    if not word.lower() in lexicons_dict and word.isalpha():
                        lexicons_dict[word.lower()] = id_counter
                        id_counter += 1

    return lexicons_dict


def encode_data(files_list, lexicons_dictionary, padding_to_length=None):
    """
    encodes the dataset into its numeric form given a constructed dictionary
    padding the vectors to the padding_to_length, by adding dummy symbols in the end
    Parameters:
    ----------
    files_list: list
        the list of files to scan through
    lexicons_dictionary: dict
        the mappings of unique lexicons
    Returns: tuple (dict, int)
        the data in its numeric form, maximum story length
    """

    files = {}
    story_inputs = None
    story_outputs = None
    stories_lengths = []
    answers_flag = False  # a flag to specify when to put data into outputs list

    limit = padding_to_length if not padding_to_length is None else float("inf")

    # add a padding symbol
    plus_index = len(lexicons_dictionary)
    lexicons_dictionary["+"] = plus_index

    for indx, filename in enumerate(files_list):

        files[filename] = []

        with open(filename, 'r') as fobj:
            for line in fobj:

                # first seperate . and ? away from words into seperate lexicons
                line = line.replace('.', ' .')
                line = line.replace('?', ' ?')
                line = line.replace(',', ' ')

                answers_flag = False  # reset as answers end by end of line

                for i, word in enumerate(line.split()):

                    if word == '1' and i == 0:
                        # beginning of a new story
                        if not story_inputs is None:
                            story_len = len(story_inputs)
                            stories_lengths.append(story_len)
                            if story_len <= limit:
                                # if below limit, padding starts
                                # input is a symbol &
                                story_inputs += [plus_index] * (limit - story_len)

                                files[filename].append({
                                    'inputs': story_inputs,
                                    'outputs': story_outputs
                                })
                        story_inputs = []
                        story_outputs = []

                    if word.isalpha() or word == '?' or word == '.':
                        if not answers_flag:
                            story_inputs.append(lexicons_dictionary[word.lower()])
                        else:
                            story_inputs.append(lexicons_dictionary['-'])
                            story_outputs.append(lexicons_dictionary[word.lower()])

                        # set the answers_flags if a question mark is encountered
                        if not answers_flag:
                            answers_flag = (word == '?')

    return files, stories_lengths


def load(path):
    return pickle.load(open(path, 'rb'))


def onehot(index, size):
    vec = np.zeros(size, dtype=np.float32)
    vec[int(index)] = 1.0
    return vec


def prepare_sample(sample, target_code, word_space_size, batch_size):
    list_of_input_vec = []
    list_of_output_vec = []
    list_of_ignore_index = []
    list_of_seq_len = []

    for i in range(batch_size):
        # input_vector is a story.
        # this is an array of ~120 elements, with hyphen
        input_vec = np.array(sample[i]['inputs'], dtype=np.float32)
        # this is an array that will have 152 elements, wihtout hyphen
        output_vec = np.array(sample[i]['inputs'], dtype=np.float32)
        seq_len = input_vec.shape[0]
        weights_vec = np.zeros(seq_len, dtype=np.float32)

        # target_mask is where an answer is required
        target_mask = (input_vec == target_code)
        critical_index = np.where(target_mask == True)
        output_vec[target_mask] = sample[i]['outputs']

        input_vec = np.array([onehot(code, word_space_size) for code in input_vec])
        #
        output_vec = output_vec.astype("long")
        # most of the output sequence is the same with the input sequence
        # except for the - part, where the machine is prompt to answer

        list_of_input_vec.append(input_vec)
        list_of_output_vec.append(output_vec)
        list_of_seq_len.append(seq_len)
        list_of_ignore_index.append(critical_index)

    input_vec = np.stack(list_of_input_vec)
    output_vec = np.stack(list_of_output_vec)
    critical_index = np.stack(list_of_ignore_index)
    critical_index = critical_index.squeeze(1)
    return (
        np.reshape(input_vec, (batch_size, -1, word_space_size)),
        output_vec,
        critical_index
    )


def write_babi_to_disk(story_limit=150):
    '''
    calls raw babi commands
    pickles train and test data
    :param task:
    :param sets:
    :param train_files_count:
    :return:
    '''
    # babi_command(task,sets,True,train=True, files_count=train_files_count)
    # babi_command(task,sets,True,train=False, files_count=int(train_files_count/5))

    task_dir = os.path.dirname(abspath(__file__))
    data_dir = join(task_dir, 'data/')
    joint_train = True
    files_list = []

    if not exists(join(task_dir, 'data')):
        mkdir(join(task_dir, 'data'))

    if data_dir is None:
        raise ValueError("data_dir argument cannot be None")

    for entryname in listdir(data_dir):
        entry_path = join(data_dir, entryname)
        if isfile(entry_path):
            files_list.append(entry_path)

    lexicon_dictionary = create_dictionary(files_list)
    lexicon_count = len(lexicon_dictionary)

    # append used punctuation to dictionary
    lexicon_dictionary['?'] = lexicon_count
    lexicon_dictionary['.'] = lexicon_count + 1
    lexicon_dictionary['-'] = lexicon_count + 2

    encoded_files, stories_lengths = encode_data(files_list, lexicon_dictionary, story_limit)

    processed_data_dir = join(task_dir, 'data',"processed")
    train_data_dir = join(processed_data_dir, 'train')
    test_data_dir = join(processed_data_dir, 'test')
    if exists(processed_data_dir) and isdir(processed_data_dir):
        rmtree(processed_data_dir)

    mkdir(processed_data_dir)
    mkdir(train_data_dir)
    mkdir(test_data_dir)

    pickle.dump(lexicon_dictionary, open(join(processed_data_dir, 'lexicon-dict.pkl'), 'wb'))

    joint_train_data = []

    for filename in encoded_files:
        if filename.endswith("test.txt"):
            pickle.dump(encoded_files[filename], open(join(test_data_dir, "test" + '.pkl'), 'wb'))
        elif filename.endswith("train.txt"):
            if not joint_train:
                pickle.dump(encoded_files[filename], open(join(train_data_dir, basename(filename) + '.pkl'), 'wb'))
            else:
                joint_train_data.extend(encoded_files[filename])

    if joint_train:
        pickle.dump(joint_train_data, open(join(train_data_dir, 'train.pkl'), 'wb'))


def gendata(batch_size, validate=False):
    '''
    The main function to generate data.
    :param batch_size:
    :param story_limit: padding the input/output vectors to length.
    :return:
    '''

    dirname = os.path.dirname(__file__)

    data_dir = os.path.join(dirname, 'data', 'processed')

    lexicon_dict = load(join(data_dir, 'lexicon-dict.pkl'))
    if validate == False:
        file_path = join(data_dir, 'train', 'train.pkl')
    else:
        file_path = join(data_dir, 'test', 'test.pkl')
    data = load(file_path)

    word_space_size = len(lexicon_dict)

    sample = np.random.choice(data, batch_size)
    input_data, target_output, ignore_index = prepare_sample(sample, lexicon_dict['-'], word_space_size, batch_size)

    # (batch_size, story_word_count, one_hot_dictionary_size)
    return input_data, target_output, ignore_index


class PreGenData():
    # the purpose of this class is to generate data before it's required to use.
    # this will reduce 11% of my code run time according to cProfiler.
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.val_ready = False
        self.train_ready = False
        self.next_train = None
        self.next_validate = None
        self.__gendata_train()
        self.__gendata_val()
        param.x = self.next_train[0].shape[2]
        param.v_t = param.x

    def get_train(self):
        Thread(target=self.__gendata_train).start()
        while not self.train_ready:
            print('train data is not ready?')
            time.sleep(1)
        return self.next_train

    def get_validate(self):
        Thread(target=self.__gendata_val).start()
        while not self.val_ready:
            print('val data is not ready?')
            time.sleep(1)
        return self.next_train

    def __gendata_train(self):
        self.next_train = gendata(self.batch_size, False)
        self.train_ready = True

    def __gendata_val(self):
        self.next_validate = gendata(self.batch_size, True)
        self.val_ready = True

def main():
    write_babi_to_disk(story_limit=150)
    pgd=PreGenData(param.bs)
    input_data, target_output, ignore_index=pgd.get_train()
    print("done")

if __name__ == '__main__':
    main()