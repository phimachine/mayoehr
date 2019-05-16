# Serialize patient data for faster loading.

from death.post.inputgen_planJ import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
def serialize(path):
    """
    The serialize function serializes each data point.
    At the moment there is no way to batch serialize them, because of how the Dataloader handles prefetch.

    :param path:
    :return:
    """
    print("This function will serialize your data at", path)
    print("This function will run quite a while, because it needs to collect every data point")
    print("There will be three progress bars in total, for training, validation and test sets")
    cache_them(path, n_proc=8)

if __name__ == '__main__':
    serialize("/local2/tmp/jasondata/")