# adapted from github.com/Mostafa-Samir/DNC-tensorflow

import sys
import pickle
import getopt
import numpy as np
from shutil import rmtree
import os
from os import listdir, mkdir
from os.path import join, isfile, isdir, dirname, basename, normpath, abspath, exists
import subprocess

def babi_command(task, sets, write_to_disk=True,train=True, files_count=1):
    # this command is run on my remote interpreter
    os.chdir("./data")
    for i in range(files_count):
        if write_to_disk:
            if train:
                babiGen="/home/jasonhu/Git/distro/install/bin/babi-tasks "+str(task)+" "+str(sets)+" babi_"+str(i)+"_train.txt"
            else:
                babiGen="/home/jasonhu/Git/distro/install/bin/babi-tasks "+str(task)+" "+str(sets)+" babi_"+str(i)+"_test.txt"
        else:
            babiGen="/home/jasonhu/Git/distro/install/bin/babi-tasks "+str(task)+" "+str(sets)

        process = subprocess.Popen(babiGen.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print('generated file '+ str(files_count))
    os.chdir("../")

def list_of_babi(task, sets):
    raw_output=babi_command(task, sets).decode("utf-8")
    raw_splitted=raw_output.split('\n')
    babi_sets=[raw_splitted[15*i:(15*i+15)] for i in range(sets)]
    return babi_sets

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
    plus_index=len(lexicons_dictionary)
    # lexicons_dictionary["+"]=plus_index
    # lexicons_dictionary["="]=plus_index+1

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
                            story_len=len(story_inputs)
                            stories_lengths.append(story_len)
                            if story_len<= limit:
                                # if below limit, padding starts
                                # input is a symbol &
                                story_inputs+=[plus_index]*(limit-story_len)
                                story_outputs+=[plus_index+1]*(limit-story_len)

                                files[filename].append({
                                    'inputs':story_inputs,
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
    list_of_input_vec=[]
    list_of_output_vec=[]
    list_of_weight_vec=[]
    list_of_seq_len=[]

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
        output_vec[target_mask] = sample[0]['outputs']
        # weights is where the hyphen is and requires an answer.
        weights_vec[target_mask] = 1.0

        input_vec = np.array([onehot(code, word_space_size) for code in input_vec])
        output_vec = np.array([onehot(code, word_space_size) for code in output_vec])
        # most of the output squence is the same with the input sequence
        # except for the - part, where the machine is prompt to answer

        list_of_input_vec.append(input_vec)
        list_of_output_vec.append(output_vec)
        list_of_seq_len.append(seq_len)
        list_of_weight_vec.append(weights_vec)

    input_vec=np.stack(list_of_input_vec)
    output_vec=np.stack(list_of_output_vec)
    seq_len=list_of_seq_len[0]
    if not all(seq_len==seq for seq in list_of_seq_len):
        raise("Sequence length not even.")
    weights_vec=np.stack(list_of_weight_vec)
    return (
        np.reshape(input_vec, (batch_size, -1, word_space_size)),
        np.reshape(output_vec, (batch_size, -1, word_space_size)),
        seq_len,
        np.reshape(weights_vec, (batch_size, -1, 1))
    )

def write_babi_to_disk(task, sets, train_files_count=1):
    babi_command(task,sets,True,train=True, files_count=train_files_count)
    babi_command(task,sets,True,train=False, files_count=int(train_files_count/5))

def datagen(batch_size,length_limit=150,padding=True):
    # batch processing has problem, because story length is uneven.

    task_dir = os.path.dirname(abspath(__file__))
    data_dir = "."
    joint_train = True
    files_list = []

    if batch_size!=1 and padding==False:
        # if batch is wanted but padding is not allowed
        raise("You must pad the input if you want to use batch.\n"
              "Story length is uneven.")

    if not exists(join(task_dir, 'data')):
        mkdir(join(task_dir, 'data'))


    if data_dir is None:
        raise ValueError("data_dir argument cannot be None")

    print(os.getcwd())
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

    encoded_files, stories_lengths = encode_data(files_list, lexicon_dictionary, length_limit)

    processed_data_dir = join(task_dir, 'data', basename(normpath(data_dir)))
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
            pickle.dump(encoded_files[filename], open(join(test_data_dir, basename(filename) + '.pkl'), 'wb'))
        elif filename.endswith("train.txt"):
            if not joint_train:
                pickle.dump(encoded_files[filename], open(join(train_data_dir, basename(filename) + '.pkl'), 'wb'))
            else:
                joint_train_data.extend(encoded_files[filename])

    if joint_train:
        pickle.dump(joint_train_data, open(join(train_data_dir, 'train.pkl'), 'wb'))

    print('start preparing sample')


    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data','data')

    lexicon_dict = load(os.path.join(data_dir, 'lexicon-dict.pkl'))
    data = load(os.path.join(data_dir, 'train', 'train.pkl'))

    word_space_size = len(lexicon_dict)

    sample = np.random.choice(data, batch_size)
    input_data, target_output, seq_len, weights = prepare_sample(sample, lexicon_dict['-'], word_space_size, batch_size)
    print("examination")

    # (batch_size, story_word_count, one_hot_dictionary_size)
    return input_data,target_output,seq_len,weights


if __name__ == '__main__':
    write_babi_to_disk(10, 1200, train_files_count=5)
    datagen(5)
