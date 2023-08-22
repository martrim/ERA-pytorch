"""Defines the Enhanced Rational Activation (ERA).
"""
import numpy as np
import torch
from torch import nn
from scipy.optimize import minpack


def get_rational_parameters(initialisation, degree_denominator, lower_bound=-3, upper_bound=3):
    print('Generating initial parameters.')
    target_functions = {'leaky': lambda x: np.maximum(x, 0.3 * x), 'relu': lambda x: np.maximum(x, 0), 'swish': lambda x: x / (1 + np.exp(-x))}

    num_weights = 2 * degree_denominator + 2
    p0 = np.ones(num_weights, dtype='float32')
    x_size = 100000
    x = np.linspace(lower_bound, upper_bound, x_size, dtype='float32')
    y = target_functions[initialisation](x)

    result = minpack.least_squares(lambda weights: era_function(
        x, weights[degree_denominator:], weights[:degree_denominator]) - y, p0, jac='3-point', method='dogbox')
    # jac='2-point'
    # method='trf'
    fitted_weights = result['x'][:, np.newaxis]
    numerator = torch.tensor(fitted_weights[degree_denominator:], dtype=torch.float32)
    denominator = torch.tensor(fitted_weights[:degree_denominator], dtype=torch.float32)
    return numerator, denominator


def era_function(x, numerator_weights, denominator_weights):
    # Computes a*x + b + sum_i (c_i * x + d_i) / ((x - e_i) ** 2 + f_i ** 2 + epsilon),
    # where i is the number of partial fractions and epsilon is a small positive number.

    output = numerator_weights[0] * x + numerator_weights[1]
    numerator_weights = numerator_weights[2:]
    epsilon = 1e-6

    num_partial_fractions = numerator_weights.shape[0] // 2
    for i in range(num_partial_fractions):
        output += (numerator_weights[2 * i] * x + numerator_weights[2 * i + 1]) / \
                  ((x - denominator_weights[2 * i]) ** 2 + denominator_weights[2 * i + 1] ** 2 + epsilon)
    return output


class ERA(torch.nn.Module):
    def __init__(self,
                 numerator,
                 denominator,
                 initialisation,
                 degree_denominator,
                 num_channels,
                 data_type='image',
                 ):
        super(ERA, self).__init__()

        # Normalisation
        if data_type == 'image':
            weight_shape = (1, num_channels, 1, 1)
            self.standardization_axes = [1, 2]
            self.learnable_axes = [3]
        elif data_type == 'tokens':
            weight_shape = (1, num_channels, 1)
            self.standardization_axes = [2]
            self.learnable_axes = []
        elif data_type == 'flat':
            weight_shape = (1, num_channels)
            self.standardization_axes = [1]
            self.learnable_axes = []  # [] or [1]
        else:
            raise ValueError('Invalid data_type.')
        self.epsilon = 1e-6
        self.beta = torch.nn.Parameter(torch.zeros(weight_shape))  # initialise with zeros
        self.gamma = torch.nn.Parameter(torch.ones(weight_shape))  # initialise with ones

        # Activation Function
        # Adding trainable weight vectors for numerator and denominator
        if initialisation == 'random':
            assert numerator is None
            assert denominator is None
            num_initializer = torch.empty(degree_denominator + 2)
            denom_initializer = torch.empty(degree_denominator)
        else:
            assert len(numerator) == degree_denominator + 2
            assert len(denominator) == degree_denominator
            num_initializer = numerator
            denom_initializer = denominator
        self.numerator = torch.nn.Parameter(num_initializer)
        self.denominator = torch.nn.Parameter(denom_initializer)

    def forward(self, x):
        # Normalisation
        mu = torch.mean(x, self.standardization_axes, keepdim=True)
        sigma = torch.std(x, self.standardization_axes, keepdim=True)
        x = self.beta + self.gamma * (x - mu) / (sigma + self.epsilon)

        # Activation Function
        return era_function(x, self.numerator, self.denominator)
