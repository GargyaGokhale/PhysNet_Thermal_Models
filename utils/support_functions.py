import torch.nn as nn

from utils.global_state_variables import MAX_TEMP, MAX_ACTION, MAX_TIME, TEMP_CONSTANT, ACTION_CONSTANT, OUTSIDE_AIR_TEMP_CONSTANT

"""Utility Functions Related To Data"""


def transform_temp(unscaled_temp):
    if type(unscaled_temp) == list:
        TypeError("Input Must be ndarray")

    scaled_temp = (unscaled_temp - TEMP_CONSTANT) / MAX_TEMP
    return scaled_temp


def inverse_transform_temp(scaled_temp):
    if type(scaled_temp) == list:
        TypeError("Input Must be ndarray")

    unscaled_temp = (scaled_temp * MAX_TEMP) + TEMP_CONSTANT
    return unscaled_temp


def transform_action(unscaled_action):
    if type(unscaled_action) == list:
        TypeError("Input Must be ndarray")

    scaled_action = (unscaled_action - ACTION_CONSTANT) / MAX_ACTION
    return scaled_action


def inverse_transform_action(scaled_action):
    if type(scaled_action) == list:
        TypeError("Input Must be ndarray")

    unscaled_action = (scaled_action * MAX_ACTION) + ACTION_CONSTANT
    return unscaled_action


def transform_outside_temp(unscaled_temp):
    if type(unscaled_temp) == list:
        TypeError("Input Must be ndarray")

    scaled_temp = (unscaled_temp - OUTSIDE_AIR_TEMP_CONSTANT) / MAX_TEMP
    return scaled_temp


def inverse_transform_outside_temp(scaled_temp):
    if type(scaled_temp) == list:
        TypeError("Input Must be ndarray")

    unscaled_temp = (scaled_temp * MAX_TEMP) + OUTSIDE_AIR_TEMP_CONSTANT
    return unscaled_temp


"""Utility Functions Related To Network"""


class fc_module(nn.Module):

    def __init__(self, layer_params, activation='tanh', dropout_rate=0.0):
        """
        :type layer_params: list[int]
        """
        super(fc_module, self).__init__()
        if activation == 'tanh':
            activation_function = nn.Tanh

        elif activation == 'relu':
            activation_function = nn.ReLU

        elif activation == 'sigmoid':
            activation_function = nn.Sigmoid

        else:
            raise ValueError(f"Unknown Activation function: {activation}")
        self.fc_module = nn.Sequential(
            nn.Linear(layer_params[0], layer_params[1]),
            activation_function(),
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, x):
        return self.fc_module(x)


def xavier_weight_initialisation(nn_module):
    if type(nn_module) == nn.Linear:
        nn.init.xavier_uniform(nn_module.weight, gain=(5 / 3))
        # nn_module.bias.data.fill_(1e-3)



