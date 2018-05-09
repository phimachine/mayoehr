# all variables starting with param refers to variables declared in this python file
# other variables are Latex representation of variables stated in Methods section of Nature paper

'''
Controller parameters
'''
# input vector size x_t
# dataset specific
x=27

# single hidden unit output size h^l_t
# state size
# output size, forget gate size, input gate size are all equal to state size s
# all weight matrices in equation 1-5 then has dimension (s, x+2*h)
# by equation 5, h=s=o
h=64

# Controller RNN layers count
# refers to the number of parallel RNN units
L=64

# Controller output v_t size
# dataset specific
v_t=32

# Memory location width
# Memory read heads count R
# Controller interface epsilon_t size, derived
W=64
R=32
E_t=W*R+3*W+5*R+3

# Total memory address count
# Total memory block (N, W)
N=32

# I am going to bake batch_processing in the DNC.
# This is going to be a very tough week.
bs=32