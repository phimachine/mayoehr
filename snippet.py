from threading import Thread
import time

class cachedYielder():
    # this function runs a value generator when it's initiated, then everytime it's called, it returns the previous value and replenishes a new one

    def __init__(self):
        self.cache=None
        self.count=0
        self.gen()

    def gen(self):
        # This is the slow generator
        time.sleep(1)
        self.cache=self.count
        self.count+=1
        print('generated and replaced')

    def get_val(self):
        Thread(target=self.gen).start()
        return self.cache

hello=cachedYielder()
while(True):
    needed=hello.get_val()
    print("needed value accessed")
    time.sleep(1)
