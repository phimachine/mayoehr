# python translate.py -model death/tran/trained.chkpt -vocab death/tran/data/multi30k.atok.low.pt -src death/tran/data/multi30k/test.en.atok

from death.tranreference.translate import main

if __name__ == '__main__':
    main()