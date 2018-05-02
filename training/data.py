import subprocess

def babi_command(task, sets):
    # this command is run on my remote interpreter
    babiGen="/home/jasonhu/Git/distro/install/bin/babi-tasks "+str(task)+" "+str(sets)
    process = subprocess.Popen(babiGen.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    if error:
        print("Error generating the dataset")
    else:
        return output

def list_of_babi(task, sets):
    raw_output=babi_command(task, sets).decode("utf-8")
    raw_splitted=raw_output.split('\n')
    babi_sets=[raw_splitted[15*i::15*i+15] for i in range(sets)]


    return 0

if __name__=="__main__":
    print(list_of_babi(10,10))