"""Defines the Enhanced Rational Activation (ERA).
"""
import numpy as np
import torch
from torch import nn
from scipy.optimize import minpack


def get_rational_parameters(initialisation, degree_denominator, lower_bound=-3, upper_bound=3):
    print('Generating initial parameters.')
    target_functions = {'leaky': nn.LeakyReLU, 'relu': nn.ReLU, 'swish': nn.SiLU, 'gelu': nn.GELU}

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
                 input_shape,
                 standardization_axes=None,
                 learnable_axes=None,
                 ):
        super(ERA, self).__init__()

        # input_shape[0]: batch_size
        # input_shape[-1]: num_channels
        assert len(input_shape) in [2, 3, 4, 5]

        self.normalise = False
        # Normalisation
        if (standardization_axes is not None) or (learnable_axes is not None):
            self.normalise = True
            self.standardization_axes = standardization_axes
            self.learnable_axes = learnable_axes
            self.epsilon = 1e-6
            self.new_shape = input_shape

            weight_shape = []
            for i in range(len(self.new_shape)):
                if i in self.learnable_axes:
                    weight_shape.append(self.new_shape[i])
                else:
                    weight_shape.append(1)
            weight_shape = tuple(weight_shape)
            self.beta = torch.nn.Parameter(torch.zeros(weight_shape))  # initialise with zeros
            self.gamma = torch.nn.Parameter(torch.ones(weight_shape))  # initialise with ones

        # Activation Function
        num_numerator_weights = degree_denominator + 2

        weight_shape_ending = [1] * len(input_shape)
        numerator_weight_shape = [num_numerator_weights] + weight_shape_ending
        denominator_weight_shape = [degree_denominator] + weight_shape_ending
        if initialisation != 'random':
            numerator = torch.repeat_interleave(numerator, weight_shape_ending[-1], dim=-1)
            denominator = torch.repeat_interleave(denominator, weight_shape_ending[-1], dim=-1)
            numerator = torch.reshape(numerator, numerator_weight_shape)
            denominator = torch.reshape(denominator, denominator_weight_shape)

        # Adding trainable weight vectors for numerator and denominator
        if initialisation == 'random':
            num_initializer = torch.empty(numerator_weight_shape)
            denom_initializer = torch.empty(denominator_weight_shape)
        else:
            num_initializer = numerator
            denom_initializer = denominator
        self.numerator = torch.nn.Parameter(num_initializer)
        self.denominator = torch.nn.Parameter(denom_initializer)

    def call(self, x):
        if self.normalise:
            # Normalisation
            mu = torch.mean(x, self.standardization_axes, keepdim=True)
            sigma = torch.std(x, self.standardization_axes, keepdim=True)
            x = self.beta + self.gamma * (x - mu) / (sigma + self.epsilon)

        # Activation Function
        return era_function(x, self.numerator, self.denominator)
