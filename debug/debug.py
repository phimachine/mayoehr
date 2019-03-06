from death.DNC.adamaxseqtrainer import *


def main():
    ig = InputGenJ(no_underlying=True, death_only=True)
    param_x=ig.input_dim
    param_v_t=ig.output_dim

    savestr="Jadam"

    total_epochs = 100
    iter_per_epoch = int(saturation/param_bs)
    optim = None
    starting_epoch = 0
    starting_iteration = 0
    logfile = "log/dnc_" + savestr + "_" + datetime_filename() + ".txt"

    computer = SeqDNC(x=param_x,
                      h=param_h,
                      L=param_L,
                      v_t=param_v_t,
                      W=param_W,
                      R=param_R,
                      N=param_N)


    print("loading model")
    computer, optim, starting_epoch, starting_iteration = load_model(computer, optim, starting_epoch,
                                                                     starting_iteration, savestr)


    binary_criterion= nn.BCEWithLogitsLoss()

    with open("itcc.pkl",'rb') as f:
        input, target, cause_of_death_output, cause_of_death_target=pickle.load(f)

    print("Done")



if __name__ == '__main__':
    main()