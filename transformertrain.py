# python train.py -data death/tran/data/multi30k.atok.low.pt -save_model trained -save_mode best -proj_share_weight -label_smoothing

from death.tran.train import main

if __name__ == '__main__':
    main()