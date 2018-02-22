# reference Methods, controller network

import archi.parameters
import torch.nn as nn

class Controller(nn.Module):
    def __init__(self):
        super(Controller, self).__init__()


    def forward(self, *input):
        """

        :param input:
        x_t: [X], input vector at time step t
        

        :return:output:
        v_t

        """
        return output, interface