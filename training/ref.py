import numpy as np
from training import data

def onehot(index, size):
    vec = np.zeros(size, dtype=np.float32)
    vec[index] = 1.0
    return vec

def prepare_sample(set_inputs, set_outputs, target_code, word_space_size):
    input_vec = np.array(set_inputs, dtype=np.float32)
    output_vec = np.array(set_inputs, dtype=np.float32)
    seq_len = input_vec.shape[0]
    weights_vec = np.zeros(seq_len, dtype=np.float32)

    target_mask = (input_vec == target_code)
    output_vec[target_mask] = set_outputs
    weights_vec[target_mask] = 1.0

    input_vec = np.array([onehot(code, word_space_size) for code in input_vec])
    output_vec = np.array([onehot(code, word_space_size) for code in output_vec])

    return (
        np.reshape(input_vec, (1, -1, word_space_size)),
        np.reshape(output_vec, (1, -1, word_space_size)),
        seq_len,
        np.reshape(weights_vec, (1, -1, 1))
    )

if __name__=="__main__":
    lob=data.list_of_babi(10,100)
    lexicon_dict=data.create_dictionary(lob)
    set_inputs, set_outputs, stories_lengths=data.encode_data(lob,lexicon_dict)
    word_space_size=len(lexicon_dict)

    prepare_sample(set_inputs, set_outputs, lexicon_dict['-'], word_space_size)