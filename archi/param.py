# all variables starting with param refers to variables declared in this python file
# other variables are Latex representation of variables stated in Methods section of Nature paper

'''
Controller parameters
'''
# input vector size x_t
x=10

# single hidden unit output size h^l_t
h=10

# state size
# output size, forget gate size, input gate size are all equal to state size s
# all weight matrices in equation 1-5 then has dimension (s, x+2*h)
s=20

# Controller RNN layers count
L=64

# Controller output v_t size
v_t=32

# Memory location width W
# Memory read heads count R
# Controller interface epsilon_t size, derived
W=5
R=7
E_t=W*R+3*W+5*R+3

# Total memory locations count
# Total memory block (n, W)
N=3