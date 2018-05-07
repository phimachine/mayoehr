# all variables starting with param refers to variables declared in this python file
# other variables are Latex representation of variables stated in Methods section of Nature paper

'''
Controller parameters
'''
# input vector size x_t
x=10

# single hidden unit output size h^l_t
# state size
# output size, forget gate size, input gate size are all equal to state size s
# all weight matrices in equation 1-5 then has dimension (s, x+2*h)
# by equation 5, h=s=o
h=10

# Controller RNN layers count
# refers to the number of parallel RNN units
L=64

# Controller output v_t size
v_t=32

# Memory location width W
# Memory read heads count R
# Controller interface epsilon_t size, derived
W=5
R=7
E_t=W*R+3*W+5*R+3

# Total memory address count
# Total memory block (N, W)
N=3

# I am going to bake batch_processing in the DNC.
# This is going to be a very tough week.
bs=16