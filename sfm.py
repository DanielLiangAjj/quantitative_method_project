import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pywt


class SFM(nn.Module):
    def __init__(self, input_size, hidden_size, freq_size, output_size):
        """
          Initialize the State Frequency Memory (SFM) module.

          Parameters:
              input_size (int): Number of input features per time step.
              hidden_size (int): Number of features in the hidden state.
              freq_size (int): Number of frequency components to model.
              output_size (int): Number of output features.
        """
        super(SFM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.freq_size = freq_size
        self.output_size = output_size

        # Activiation Functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # Output layer
        self.output = nn.Linear(hidden_size, output_size)

        # Precompute omega as a constant, based on frequency size
        self.omega = Variable(torch.tensor(2 * np.pi * np.arange(1, self.freq_size + 1) / self.freq_size).float())

        # Initialize parameters
        self.init_parameters()
        self.reset_parameters()

    def reset_parameters(self):
        """
            Reset parameters with a uniform distribution based on hidden size, improving initialization.
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def init_parameters(self):
        """
            Initialize weights and biases for different gates and transformations within the SFM.
        """
        # gate weights
        # W -> x, U -> h_last

        # Input gate weights and biases
        self.i_W = Parameter(torch.randn(self.input_size, self.hidden_size))
        self.i_U = Parameter(torch.randn(self.hidden_size, self.hidden_size))
        self.i_b = Parameter(torch.randn(self.hidden_size))

        # Candidate gate weights and biases
        self.g_W = Parameter(torch.randn(self.input_size, self.hidden_size))
        self.g_U = Parameter(torch.randn(self.hidden_size, self.hidden_size))
        self.g_b = Parameter(torch.randn(self.hidden_size))

        # Output gate weights and biases
        self.o_W = Parameter(torch.randn(self.input_size, self.hidden_size))
        self.o_U = Parameter(torch.randn(self.hidden_size, self.hidden_size))
        self.o_V = Parameter(torch.randn(self.hidden_size, self.hidden_size))
        self.o_b = Parameter(torch.randn(self.hidden_size))

        # Frequency and state transformation weights and biases
        self.fre_W = Parameter(torch.randn(self.input_size, self.freq_size))  # belong R^K
        self.fre_U = Parameter(torch.randn(self.hidden_size, self.freq_size))
        self.fre_b = Parameter(torch.randn(self.freq_size))

        self.ste_W = Parameter(torch.randn(self.input_size, self.hidden_size))
        self.ste_U = Parameter(torch.randn(self.hidden_size, self.hidden_size))
        self.ste_b = Parameter(torch.randn(self.hidden_size))

        # Amplitude computation weights and biases
        self.a_U = Parameter(torch.randn(self.freq_size, 1))
        self.a_b = Parameter(torch.randn(self.hidden_size))


    def forward(self, input, h, c, re_s, im_s, time):
        """
           Forward pass of the SFM module.

           Parameters:
               input (Tensor): Input tensor at current time step.
               h (Tensor): Hidden state.
               c (Tensor): Cell state.
               re_s (Tensor): Real part of the state frequency component.
               im_s (Tensor): Imaginary part of the state frequency component.
               time (Tensor): Time step variable.

           Returns:
               tuple: Contains output, updated hidden state, cell state, real and imaginary components, and time.
       """
        # Input and candidate gates
        i_t = self.sigmoid(torch.matmul(input, self.i_W) + torch.matmul(h, self.i_U) + self.i_b)
        c_hat_t = self.sigmoid(torch.matmul(input, self.g_W) + torch.matmul(h, self.g_U) + self.g_b)

        # Frequency and state transformations
        f_ste = self.sigmoid(torch.matmul(input, self.ste_W) + torch.matmul(h, self.ste_U) + self.ste_b)  # belong R^D
        f_fre = self.sigmoid(torch.matmul(input, self.fre_W) + torch.matmul(h, self.fre_U) + self.fre_b)  # belong R^K
        f_t = torch.matmul(f_ste.view(-1, self.hidden_size, 1), f_fre.view(-1, 1, self.freq_size))

        # exp_damping = torch.exp(-time.pow(2) / 2)
        # pi_coefficient = np.pi ** (-0.25)
        cos_part = torch.cos(self.omega * time).unsqueeze(0)
        sin_part = torch.sin(self.omega * time).unsqueeze(0)

        # Update the real and imaginary parts of the frequency state
        re_s = torch.mul(f_t, re_s) + torch.matmul(torch.mul(i_t, c_hat_t).transpose(1, 2), cos_part)
        im_s = torch.mul(f_t, im_s) + torch.matmul(torch.mul(i_t, c_hat_t).transpose(1, 2), sin_part)
        # re_s = torch.mul(f_t, re_s) + torch.matmul(torch.mul(i_t, c_hat_t).transpose(1, 2),
        #                                            torch.cos(torch.mul(self.omega, time)).unsqueeze(0))
        #         * pi_coefficient *exp_damping)
        # im_s = torch.mul(f_t, im_s) + torch.matmul(torch.mul(i_t, c_hat_t).transpose(1, 2),
        #                                            torch.sin(torch.mul(self.omega, time)).unsqueeze(0))
        #         * pi_coefficient *exp_damping)

        # Calculate the amplitude (A_t) using the updated real and imaginary parts
        a_t = torch.sqrt(re_s ** 2 + im_s ** 2)
        # Combine the amplitude with other transformations to update the cell state
        c_t = self.tanh(torch.matmul(a_t, self.a_U).transpose(1, 2) + self.a_b)

        # Output gate and final output computation
        o_t = self.sigmoid(
            torch.matmul(input, self.o_W) + torch.matmul(h, self.o_U) + torch.matmul(c_t, self.o_V) + self.o_b)
        h_t = torch.mul(o_t, self.tanh(c_t))

        # Compute final output vector for the current timestep
        output = self.output(h_t)
        output = output.view(-1, 1)
        # Increment time for the next forward call
        time += 1
        return output, h_t, c_t, re_s.squeeze(0), im_s.squeeze(0), time

    def init_state(self):
        """
            Initializes the states required for the SFM computation: hidden state, cell state,
            real and imaginary parts of the frequency state, and time.

            Returns:
                tuple: Initial states for h, c, re_s, im_s, and time.
        """
        h = Variable(torch.zeros(1, self.hidden_size))
        c = Variable(torch.zeros(1, self.hidden_size))
        re_s = Variable(torch.zeros(self.hidden_size, self.freq_size))
        im_s = Variable(torch.zeros(self.hidden_size, self.freq_size))
        time = Variable(torch.ones(1))
        return h, c, re_s, im_s, time


