from death.tranreference.preprocess import main

if __name__ == '__main__':
    main()

"""
python transformerprep.py -train_src death/tran/data/multi30k/train.en.atok -train_tgt death/tran/data/multi30k/train.de.atok -valid_src death/tran/data/multi30k/val.en.atok -valid_tgt death/tran/data/multi30k/val.de.atok -save_data death/tran/data/multi30k.atok.low.pt
"""