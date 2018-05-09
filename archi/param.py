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


'''
Hyperarameters have been tuned for Nvidia 1080. Bottleneck by memory.
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 1080    Off  | 00000000:0B:00.0  On |                  N/A |
| 47%   54C    P2    68W / 180W |   6511MiB /  8116MiB |     45%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1400      G   /usr/lib/xorg/Xorg                            40MiB |
|    0      1441      G   /usr/bin/gnome-shell                          49MiB |
|    0      1622      G   /usr/lib/xorg/Xorg                           727MiB |
|    0      1765      G   /usr/bin/gnome-shell                         529MiB |
|    0      5210      G   ...-token=3A0C47659340A06C9B73AE9394F12BB2   135MiB |
|    0      5832      G   ...passed-by-fd --v8-snapshot-passed-by-fd   124MiB |
|    0      9914      G   ...-token=C26BA1388B5B5BDF6B6980F03E2EB7E0   106MiB |
|    0     19213      C   ...onhu/anaconda3/envs/python36/bin/python  4783MiB |
+-----------------------------------------------------------------------------+
'''