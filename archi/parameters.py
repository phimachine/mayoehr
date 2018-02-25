# all variables starting with param refers to variables declared in this python file
# other variables are Latex representation of variables stated in Methods section of Nature paper

'''
Controller parameters
'''
# input vector size x_t
param_x=10

# single hidden unit output size h^l_t
param_h=10

# state size
# output size, forget gate size, input gate size are all equal to state size param_s
# all weight matrices in equation 1-5 then has dimension (param_s, param_x+2*param_h)
param_s=20

# Controller RNN layers count
param_L=64

# Controller output v_t size
param_v_t=32

# Controller interface epsilon_t size
param_E_t=32