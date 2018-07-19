class Foo():
    def __init__(self):
        a="hello"
        self.__getattribute__(a)=None

