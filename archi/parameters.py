# all variables starting with param refers to variables declared in this python file
# other variables are Latex representation of variables stated in Methods section of Nature paper

'''
Controller parameters
'''
# input vector size x_t
param_x=10

# single hidden unit size h^l_t
param_h=10

# state size
# output size, forget gate size, input gate size are all equal to state size param_s
# all weight matrices in equation 1-5 then has dimension (param_s, param_x+2*param_h)
param_s=20


# RNN layers count
param_L=64
