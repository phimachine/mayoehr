
import os
import csv
from pathlib import Path

def strip(words):
    new_words=[]
    for word in words:
        word=word.strip('.,:')
        new_words.append(word)
    return new_words

def isnum(s):
    try:
        int(s)
        return True
    except ValueError:
        try:
            float(s)
            return True
        except ValueError:
            return False

def print_file(path):
    with path.open('r') as f:
        for line in f:
            print(line)

class LogCollector():
    def __init__(self, log_path, min_epoch=9):
        self.log_path=log_path
        self.min_epoch=min_epoch

    def get_long_logs(self):
        """
        longer logs usually contain data
        short logs often are just test runs
        :return:
        """
        models = {}
        maxfiles = {}

        for lp in self.log_path:
            lp=Path(lp)
            for file in lp.iterdir():
                model=file.name.split("_")[0]
                if model in models:
                    models[model].append(file)
                else:
                    models[model]=[file]

        for model in models:
            files=models[model]
            max_size=0
            max_file=None
            for file in files:
                size=file.stat().st_size
                if size>max_size:
                    max_size=size
                    max_file=file

            # check max file
            with max_file.open("r") as f:
                okay=False
                for line in f:
                    words=line.split()
                    words=strip(words)
                    for i in range(len(words)):
                        if words[i]=="epoch":
                            epo=words[i+1]
                            if int(epo)>self.min_epoch:
                                okay=True
                                break
            if okay:
                maxfiles[model]=[max_file]

        return maxfiles


class Plotter():
    def __init__(self, files):
        """

        :param files: a dictionary, {"model":[path, path]} or a single path
        """
        self.files=files
        for model in self.files:
            self.files[model]=[Path(f) for f in self.files[model]]

        csv_dir_path=os.path.dirname(os.path.abspath(__file__))
        csv_dir_path=Path(csv_dir_path)/"csv"
        csv_dir_path.mkdir(exist_ok=True)
        self.csv_dir_path=csv_dir_path



    def manually_enter_files(self, model_files):
        """

        :return:
        """

        self.model_files=model_files

    def train_init_columns(self, words):
        try:
            words.remove("running")
        except ValueError:
            pass
        for i in range(len(words)):
            if i % 2 == 0:
                assert (words[i].isalpha())
            else:
                assert (isnum(words[i]))
        for i in range(len(words)):
            if i % 2 == 0:
                coln = words[i]
                self.cols.append(coln)
                self.__setattr__(coln, [])
            else:
                pass

    def valid_init_columns(self,words):
        for i in range(len(words)):
            if i % 2 == 0:
                assert (words[i].isalpha())
            else:
                assert (isnum(words[i]))
        for i in range(len(words)):
            if i % 2 == 0:
                coln = words[i]
                self.cols_valid.append(coln)
                self.__setattr__(coln + "_valid", [])
            else:
                pass
    def replace_model_name(self,model_name):
        replace={"softmaxDNC": "priorDNC",
                 "softmaxADNC": "APDNC"}
        if model_name in replace:
            model_name = replace[model_name]
        return model_name

    def init_columns(self, file, model_name):
        model_name=self.replace_model_name(model_name)
        self.cols=[]
        self.cols_valid = []

        with open(file, 'r') as f:
            while True:
                first_line = f.readline()
                if first_line:
                    words=first_line.split()
                    if words[0]==model_name:
                        break
                else:
                    raise ValueError("Nothing read for this file")

            words = strip(words)
            if "train" in words:
                words.remove("train")
            self.train_init_columns(words[1:])
            line=f.readline()
            words = strip(line.split())
            assert(words[1]=="train")
            self.train_init_columns(words[2:])

            line=f.readline()
            words = strip(line.split())
            assert(words[1]=="validation")
            self.valid_init_columns(words[2:])

            line=f.readline()
            words = strip(line.split())
            assert words[1] == "validate"
            self.valid_init_columns(words[2:])

            self.epoch_valid=[]
            self.cols_valid=["epoch"]+self.cols_valid


    def push_train(self, words):
        try:
            words.remove("running")
        except ValueError:
            pass
        try:
            words.remove("train")
        except ValueError:
            pass
        for i in range(len(words) // 2):
            assert (words[i * 2].isalpha())
            col = words[i * 2]
            thelist = self.__getattribute__(col)
            thelist.append(words[i * 2 + 1])

    def push_valid(self, words):
        try:
            words.remove("validation")
        except ValueError:
            pass

        try:
            words.remove("validate")
        except ValueError:
            pass


        for i in range(len(words) // 2):
            assert (words[i * 2].isalpha())
            col = words[i * 2]
            thelist = self.__getattribute__(col + "_valid")
            thelist.append(words[i * 2 + 1])

    def push_line(self,model,line):
        model=self.replace_model_name(model)

        words = line.split()
        words = strip(words)
        if len(words) < 2:
            pass
        elif words[0] != model:
            pass
        elif words[1] == "epoch":
            self.push_train(words)
        elif words[1] == "train":
            words = words[1:]
            self.push_train(words)
        elif words[1] == "validation":
            self.epoch_valid.append(self.current_epoch)
            words = words[1:]
            self.push_valid(words)
        elif words[1] == "validate":
            words = words[1:]
            self.push_valid(words)
        else:
            pass

    def write_csv_files(self, model_name, trcsvname, validcsvname):
        n = len(self.__getattribute__(self.cols[0]))
        for col in [self.__getattribute__(col) for col in self.cols]:
            assert (len(col) == n)
        with (trcsvname).open('w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(self.cols)
            for i in range(n):
                row = [self.__getattribute__(col)[i] for col in self.cols]
                writer.writerow(row)

        n = len(self.cod_valid)
        with (validcsvname).open('w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(self.cols_valid)
            for i in range(n):
                row = [self.__getattribute__(col + "_valid")[i] for col in self.cols_valid]
                writer.writerow(row)

        print(model_name, "is saved at", trcsvname, "and", validcsvname)

    def make_csv(self):
        for model, files in self.files.items():
            for i in range(len(files)):
                file=files[i]
                if i==0:
                    self.init_columns(file,model)
                    with file.open('r') as f:
                        for line in f:
                            lineepoch = self.get_epoch(line)
                            if lineepoch is not None:
                                self.current_epoch=lineepoch
                            self.push_line(model,line)
                if i!=0:
                    with file.open('r') as f:
                        skipping = True
                        for line in f:
                            lineepoch=self.get_epoch(line)
                            if lineepoch is not None:
                                self.current_epoch=lineepoch
                            if lineepoch is not None and lineepoch>max([int(epoch) for epoch in self.epoch]):
                                skipping=False
                            if not skipping:
                                self.push_line(model,line)

            trcsvname = self.csv_dir_path / (model + "_train" + ".csv")
            validcsvname = self.csv_dir_path / (model + "_valid" + ".csv")

            self.write_csv_files(model,trcsvname,validcsvname)

    def assert_same_length(self):
        n = len(self.__getattribute__(self.cols[0]))
        for col in [self.__getattribute__(col) for col in self.cols]:
            assert (len(col) == n)

    def get_epoch(self,line):
        words=line.split()
        words=strip(words)
        epoch=None
        for i in range(len(words)):
            if words[i]=="epoch":
                epoch=int(words[i+1])
        return epoch

    def get_max_epoch(self):
        return max(self.epoch_train)

def plot_manually():
    # lc=LogCollector(["/home/m193194/local2/new_jason_project/death/unified/log",
    #              "/home/m193194/local2/new_jason_project/log"])
    # files=lc.get_long_logs()
    dir="/home/m193194/local2/new_jason_project/log/"
    files={"priorlstm":["priorlstm_05_15_21:07:37.txt",
                        "priorlstm_05_15_22:23:40.txt",
                        "priorlstm_05_15_23:03:29.txt"],
           "priortaco":["priortaco_05_15_21:14:05.txt",
                        "priortaco_05_15_22:30:03.txt",
                        "taco_priortaco_05_18_22:21:29.txt",
                        "taco_priortaco_05_19_02:34:26.txt",
                        "taco_priortaco_05_19_21:00:00.txt"],
           "priorDNC":["dnc_rerun_priorDNC_05_16_00:31:42.txt",
                       "dnc_rerun_priorDNC_05_16_02:15:03.txt",
                       "dnc_rerun_priorDNC_05_19_02:34:23.txt",
                       "dnc_rerun_priorDNC_05_19_17:39:19.txt",
                       "dnc_rerun_priorDNC_05_20_23:39:04.txt"],
           "APDNC": ["dnc_rerun_APDNC_05_15_23:42:37.txt",
                     "dnc_rerun_APDNC_05_16_02:15:01.txt",
                     "adnc_rerun_APDNC_05_19_02:45:57.txt",
                     "adnc_rerun_APDNC_05_19_22:04:28.txt"],
           "tranforwardsoftmax":["tranforwardsoftmax_tranforwardsoftmax_05_16_07:49:00.txt"],
           "tranattnsoftmax":["tranattnsoftmax_tranattnsoftmax_05_18_15:57:11.txt"],
           "tranmixedforwardsoftmax":["tranmixedforwardsoftmax_tranmixedforwardsoftmax_05_16_15:07:14.txt"],
           "tranmixedattnsoftmax":["tranmixedattnsoftmax_tranmixedattnsoftmax_05_16_15:07:14.txt"],
           "softmaxADNC":["dnc_adnc_softmax_APDNC_05_17_13:28:52.txt",
                          "dnc_adnc_softmax_APDNC_05_19_05:46:23.txt",
                          "dnc_adnc_softmax_APDNC_05_20_01:57:07.txt"],
           "softmaxDNC":["dnc_adnc_softmax_priorDNC_05_17_13:28:57.txt",
                         "dnc_adnc_softmax_priorDNC_05_19_05:46:29.txt",
                         "dnc_adnc_softmax_priorDNC_05_20_01:57:13.txt"],
           "ADNCMEM":["adncvariations2_ADNCMEM_05_16_16:38:19.txt",
                      "adncvariations2_ADNCMEM_05_19_13:26:01.txt",
                      "adncvariations2_ADNCMEM_05_20_10:11:04.txt"],
           "ADNCDrop":["adncvariations2_ADNCDrop_05_16_16:38:19.txt",
                       "adncvariations2_ADNCDrop_05_19_13:26:01.txt",
                       "adncvariations2_ADNCDrop_05_20_10:11:04.txt"],
           "ADNCNorm":["adncvariations1_ADNCNorm_05_17_02:15:26.txt",
                       "adncvariations1_ADNCNorm_05_19_05:09:30.txt",
                       "adncvariations1_ADNCNorm_05_19_21:38:18.txt"],
           "ADNCbi":["adncvariations1_ADNCbi_05_17_02:15:26.txt",
                     "adncvariations1_ADNCbi_05_19_05:09:30.txt",
                     "adncvariations1_ADNCbi_05_19_21:38:18.txt"],
           "DNC":["DNC_05_14_21:24:55.txt",
                  "prior_ablation_DNC_05_18_18:10:06.txt",
                  "prior_ablation_DNC_05_19_13:15:51.txt",
                  "prior_ablation_DNC_05_20_06:26:30.txt"],
           "simple":["simple_05_15_01:41:06.txt"]
           }
    for model in files:
        files[model]=[dir+path for path in files[model]]
    plot=Plotter(files)
    plot.make_csv()


if __name__ == '__main__':
    plot_manually()

# to be rerun:
# priorLSTM: DONE
# priorTaco: DONE
# ADNC: DONE
# DNC: DONE
# DNC no prior:
# Transformer mixed forward (dense) DONE
# Transformer mixed attn: DONE
# Transformer forward (dense): DONE
# Transformer attn: DONE
# softmax DNC DONE
# softmax ADNC DONE
# 4 ADNC variants DONE
# simple: DONE