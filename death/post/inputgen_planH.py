# Plan H caches InputGenG
# instead of loading pandas pickles, it loads numpy pickles directly.
# I traded 1TB storage space for computation speed. This is worth it definitely.
# InputGenG will still be available in the object. However, the heavy lifting work is not done by its __getitem__()
# InputGenH is very rigid. It does not allow any __init__() parameters, because almost any change would require
# recaching the whole InputGenG dataset.


from death.post.inputgen_planG import *

class InputGenH():

    def __init__(self):
        self.dspath="/local2/tmp/jasondata/zerofold/"
        self.inputgenG=InputGenG(death_fold=0)
        print("G initiated")

    def get_valid(self):
        return DatasetCacher(id="valid", dataset=self.inputgenG.get_valid(), path=self.dspath, max=5000)

    def get_test(self):
        return DatasetCacher(id="test", dataset=self.inputgenG.get_test(), path=self.dspath,max=5000)

    def get_train(self):
        return DatasetCacher(id="train", dataset=self.inputgenG.get_train(), path=self.dspath, max=50000)


def target_investigation():
    ig=InputGenH()
    valid=ig.get_valid()
    for i in range(100):
        input, target, loss_type=valid[i]
        print(target[0])

def main():
    ig=InputGenH()
    valid=ig.get_valid()
    print(valid[4])

if __name__=="__main__":
    target_investigation()