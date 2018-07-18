


# all variables starting with param refers to variables declared in this python file
# other variables are Latex representation of variables stated in Methods section of Nature paper

# input vector size x_t
# dataset specific, you need to change it every time.
x=47781

# single hidden unit output size h^l_t
# state size
# output size, forget gate size, input gate size are all equal to state size s
# all weight matrices in equation 1-5 then has dimension (s, x+2*h)
# by equation 5, h=s=o
h=2

# Controller RNN layers count
# refers to the number of parallel RNN units
L=3

# Controller output v_t size
# dataset specific, you need to change it every time.
v_t=3654

# Memory location width
# Memory read heads count R
# Controller interface epsilon_t size, derived
W=4
R=5
E_t=W*R+3*W+5*R+3

# Total memory address count
# Total memory block (N, W)
N=6

# bake batch_processing in the DNC.
bs=1
