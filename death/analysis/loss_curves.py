# There is an overfitting issue.
# I want to know which epoch it started happening, and if I should train further.
# The architecture has not saturated at training time.

training_cod=[]
training_toe=[]
training_tt=[]
training_sen=[]
training_spe=[]
training_roc=[]


valid_cod=[]
valid_toe=[]
valid_tt=[]
valid_sen=[]
valid_spe=[]
valid_roc=[]
epochs=[]


files=[("adnclog.txt","adnc.csv")]

with open("adnclog.txt",'r') as f:
    while True:
        line=f.readline()
        if not line:
            break
        words=line.split()
        if words[3] == "batch":
            if words[4] == "0.":
                # this is a representation of the epoch
                # I just care about running
                cod=words[7][:-1]
                toe=words[9][:-1]
                tt=words[11]

                line=f.readline()
                words=line.split()
                sen=words[3][:-1]
                spe=words[5][:-1]
                roc=words[7][:-1]

                training_cod.append(cod)
                training_toe.append(toe)
                training_tt.append(tt)
                training_sen.append(sen)
                training_spe.append(spe)
                training_roc.append(roc)

                #print("STop here where")

        if words[0] == "model":
            if words[2] =="for":
                epoch=words[4]
                epochs.append(epoch)
        if words[1] == "validation.":
            cod=words[3][:-1]
            toe=words[5][:-1]
            total=words[7]

            valid_cod.append(cod)
            valid_toe.append(toe)
            valid_tt.append(total)
            #print("Stop me here")


        if words[1] == "validate":
            sen=words[3][:-1]
            spe=words[5][:-1]
            roc=words[7]

            valid_sen.append(sen)
            valid_spe.append(spe)
            valid_roc.append(roc)

            # this is a set of data.
            #print("Stop me here")
assert(len(training_cod)==len(training_spe))
assert(len(training_tt)==len(valid_cod))
# assert(len(valid_spe)==len(valid_cod))

with open("adnc.csv","w") as f:
    f.write("epoch, tcod, ttoe, ttt, tsen, tspe, troc, vcod, vtoe, vtt, vsen, vspe, vroc\n")
    for i in range(len(training_cod)):

        try:
            strs=[]
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

            newline=", ".join(strs)
            f.write(newline+"\n")
            #print("Stop")
        except IndexError:
            pass

    f.write("\n")

print("Done")