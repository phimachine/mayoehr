# this script processes the log files en masse
# the output is a .csv file with all information. the file name is a unique id for the model and train
# this is very awkwardly programmed and obsolete
# use plotter.py

from pathlib import Path
import os
from os.path import abspath

rootdir = Path(os.path.dirname(abspath(__file__))).parent

class LogEntry():
    """
    Usually I use tuples. What about objects?
    """
    def __init__(self, fpath, model_name):
        self.fpath=fpath
        self.model_name=model_name

def process_one_file(logentry, rootdir):
    csv_dir=rootdir/"csvs"
    if not csv_dir.exists():
        csv_dir.mkdir(exist_ok=True)

    training_cod = []
    training_toe = []
    training_tt = []
    training_sen = []
    training_spe = []
    training_roc = []

    valid_cod = []
    valid_toe = []
    valid_tt = []
    valid_sen = []
    valid_spe = []
    valid_roc = []
    epochs = []

    with open(logentry.fpath, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            words = line.split()
            if len(words)<4:
                pass
            else:
                # print(line)
                if words[3] == "batch":
                    epoch=words[2][:-1]
                    epochs.append(epoch)

                    if words[4] == "0.":
                        # this is a representation of the epoch
                        # I just care about running
                        cod = words[7][:-1]
                        toe = words[9][:-1]
                        tt = words[11]

                        line = f.readline()
                        words = line.split()
                        sen = words[3][:-1]
                        spe = words[5][:-1]
                        roc = words[7][:-1]

                        training_cod.append(cod)
                        training_toe.append(toe)
                        training_tt.append(tt)
                        training_sen.append(sen)
                        training_spe.append(spe)
                        training_roc.append(roc)

                        # print("STop here where")

                # if words[0] == logentry.model_name:
                #     if words[1] == "epoch":
                #         epoch = words[2][:-1]
                #         epochs.append(epoch)
                if words[1] == "validation.":
                    cod = words[3][:-1]
                    toe = words[5][:-1]
                    total = words[7]

                    valid_cod.append(cod)
                    valid_toe.append(toe)
                    valid_tt.append(total)
                    # print("Stop me here")

                if words[1] == "validate":
                    sen = words[3][:-1]
                    spe = words[5][:-1]
                    roc = words[7]

                    valid_sen.append(sen)
                    valid_spe.append(spe)
                    valid_roc.append(roc)

                    # this is a set of data.
                    # print("Stop me here")
    csv_file=csv_dir/(logentry.model_name+".csv")
    if csv_file.exists():
        csv_file.unlink()
    with open(csv_file, "w") as f:
        f.write("epoch, tcod, ttoe, ttt, tsen, tspe, troc, vcod, vtoe, vtt, vsen, vspe, vroc\n")
        count=0
        for i in range(len(training_cod)):

            try:
                strs = []
                strs.append(epochs[i])
                strs.append(training_cod[i])
                strs.append(training_toe[i])
                strs.append(training_tt[i])
                strs.append(training_sen[i])
                strs.append(training_spe[i])
                strs.append(training_roc[i])

                strs.append(valid_cod[i])
                strs.append(valid_toe[i])
                strs.append(valid_tt[i])
                strs.append(valid_sen[i])
                strs.append(valid_spe[i])
                strs.append(valid_roc[i])

                newline = ", ".join(strs)
                f.write(newline + "\n")
                # print("Stop")
                count+=1
            except IndexError:
                pass

        print("Wrote to file", csv_file, "," , count, "lines total")
        f.write("\n")

def mass_process(rootdir):
    logdir=rootdir/"somelogs"

    # these are the successful runs that collected data that I want
    interested_time=["03_26_21_33_00.txt","03_20_17_51_44.txt","03_26_13_52_14.txt",
                     "03_20_00_17_27.txt","03_29_18_09_49.txt","03_26_13_52_15.txt",
                     "03_31_17_09_13.txt","03_26_21_32_59.txt","04_02_19_48"]
    # interested_time=["03_20_00_17_27.txt"]
    wanted_files=[]
    for file in logdir.iterdir():
        name=file.name
        for time in interested_time:
            if time in name:
                wanted_files.append(file)

    entries=[]
    for file in wanted_files:
        name=file.name
        model=name.split("_")[0]
        entries.append(LogEntry(file, model))

    count=0
    for entry in entries:
        count+=1
        print(count)
        process_one_file(entry,rootdir)


if __name__ == '__main__':
    mass_process(rootdir)