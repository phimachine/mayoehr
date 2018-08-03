# all variables starting with param refers to variables declared in this python file
# other variables are Latex representation of variables stated in Methods section of Nature paper

'''
Controller parameters
'''

class ParamManager():
    # make it an object so I can modify it on the fly.

    def __init__(self):

        # input vector size x_t
        # dataset specific
        self.x=None

        # single hidden unit output size h^l_t
        # state size
        # output size, forget gate size, input gate size are all equal to state size s
        # all weight matrices in equation 1-5 then has dimension (s, x+2*h)
        # by equation 5, h=s=o
        self.h=128

        # Controller RNN layers count
        # refers to the number of parallel RNN units
        self.L=64

        # Controller output v_t size
        # dataset specific
        self.v_t=None

        # Memory location width
        # Memory read heads count R
        # Controller interface epsilon_t size, derived
        self.W=64
        self.R=32
        self.E_t=W*R+3*W+5*R+3

        # Total memory address count
        # Total memory block (N, W)
        self.N=32

        # bake batch_processing in the DNC.
        self.bs=1

param=ParamManager()