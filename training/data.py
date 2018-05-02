import subprocess
import signal
import os

def babi_command(task, sets):
    # this command is run on my remote interpreter
    babiGen="/home/jasonhu/Git/distro/install/bin/babi-tasks "+str(task)+" "+str(sets)
    process = subprocess.Popen(babiGen.split(), stdout=subprocess.PIPE, preexec_fn=os.setsid)
    output, error = process.communicate()
    if error:
        process.kill()
        print("Error generating the dataset")
    else:
        process.kill()
        return output

def list_of_babi(task, sets):
    raw_output=babi_command(task, sets).decode("utf-8")
    raw_splitted=raw_output.split('\n')
    babi_sets=[raw_splitted[15*i:(15*i+15)] for i in range(sets)]
    return babi_sets

def create_dictionary(babi_sets):
    lexicons_dict={}
    id_counter=0

    for set in babi_sets:
        for line in set:
            line = line.replace('.', ' .')
            line = line.replace('?', ' ?')
            line = line.replace(',', ' ')

            for word in line.split():
                if not word.lower() in lexicons_dict and not word.isdigit():
                    lexicons_dict[word.lower()] = id_counter
                    id_counter += 1
    lexicons_dict["-"]=id_counter
    return lexicons_dict

def encode_data(babi_sets, lexicons_dictionary):

    story_inputs = None
    story_outputs = None
    stories_lengths = []

    set_inputs = []
    set_outputs = []

    for story in babi_sets:

        for line in story:

            # first seperate . and ? away from words into separate lexicons
            line = line.replace('.', ' .')
            line = line.replace('?', ' ?')
            line = line.replace(',', ' ')

            answers_flag = False  # reset as answers end by end of line

            for i, word in enumerate(line.split()):

                if word == '1' and i == 0:
                    # beginning of a new story
                    if story_inputs!=None:
                        set_inputs.append(story_inputs)
                        set_outputs.append(story_outputs)

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

    print ("\rEncoding Data ... Done!")
    return set_inputs, set_outputs, stories_lengths

if __name__=="__main__":
    lob=list_of_babi(10,1000)
    dit=create_dictionary(lob)
    set_inputs, set_outputs, stories_lengths=encode_data(lob,dit)
