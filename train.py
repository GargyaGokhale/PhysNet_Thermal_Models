"""
Training script for PhysNet, PhysRegMLP models

Date: 25-03-2022

Author: Gargya Gokhale

"""
import warnings
import logging

import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl

from utils.support_functions import transform_temp, transform_action, transform_outside_temp
from utils.global_state_variables import MAX_TIME

from PhysNet import PhysNet as PhysNet
from PhysRegMLP import PhysRegMLP as PhysRegMLP

# To stop Pytorch Lightning from giving GPU Availability and warning messages
warnings.filterwarnings('ignore')
logging.getLogger('lightning').setLevel(0)

# -------------------------------------------------------------------------------------------------------------------- #


def prepare_data(data):
    x_inputs = data['x_agg_k']
    y_labels = data['label_k']

    # Check if horizon_size of inputs and outputs is the same
    if x_inputs.shape[0] != y_labels.shape[0]:
        raise ValueError("Size of inputs and labels should be the same")

    state_depth = (x_inputs.shape[1] - 4) // 2
    # Scaling
    x_inputs[:, 0] = x_inputs[:, 0] / MAX_TIME  # Time
    x_inputs[:, 1] = transform_temp(x_inputs[:, 1])  # Current Temp
    x_inputs[:, 2:(2 + state_depth)] = transform_temp(x_inputs[:, 2:(2 + state_depth)])  # Previous temp states
    x_inputs[:, (2 + state_depth):(2 + 2 * state_depth)] = transform_action(
        x_inputs[:, (2 + state_depth):(2 + 2 * state_depth)])  # Previous Actions
    x_inputs[:, -2] = transform_action(x_inputs[:, -2])  # Current Action
    x_inputs[:, -1] = transform_outside_temp(x_inputs[:, -1])  # Outside Temp

    y_labels[:, 0] = transform_temp(y_labels[:, 0])
    y_labels[:, 1] = transform_action(y_labels[:, 1])

    return {"x_agg_k": np.array(x_inputs[0:-1]),
            "label_k": np.array(y_labels[0:-1]),
            "x_agg_k1": np.array(x_inputs[1:]),
            "label_k1": np.array(y_labels[1:])
            }


def prepare_test_data(test_data):
    temp_track = []
    building_mass_temp_track = []
    time_track = []
    plotting_time_track = []
    outside_temperature = []
    u_phys_track = []
    u_track = []
    day_i = 0
    quarter_i = 0
    for index in range(test_data.shape[0]):
        temp_track.append(test_data[index, 1])
        time_track.append(test_data[index, 0])

        u_phys_track.append(test_data[index, -2])
        u_track.append(test_data[index, 2])
        outside_temperature.append(test_data[index, -1])
        plotting_time_track.append(day_i * 24 + (quarter_i * 30) / 60)

        quarter_i += 1
        if quarter_i % 48 == 0:
            day_i += 1
            quarter_i = 0
    # ------------------------------------------------------------------------------------------------------------ #
    return {
        'temp_track': temp_track,
        'building_mass_temp_track': building_mass_temp_track,
        'time_track': time_track,
        'plotting_time_track': plotting_time_track,  # Same time for different day is differentiated
        'u_phys_track': u_phys_track,
        'u_track': u_track,
        'outside_temperature': outside_temperature
    }


if __name__ == '__main__':

    # Simulation Params
    depth = 8
    lambda_2 = 0.5
    model_type = 'PhysRegMLP'           # PhysNet or PhysRegMLP
    model_seed = 1

    # -----------------------------------------------------------------------------------------------------------------#

    if model_type == 'PhysRegMLP':
        model = PhysRegMLP
        network_param = {
            'lr': 0.001,
            'batch_size': 2048,
            'lambda_value': lambda_2,
            'mdp_network': {'input_size': 4 + 2 * (depth + 0),  # [time, x_k, prev_state, prev_action, u_k, Ta_k]
                            'fc': [64] * 2,
                            'output_size': 3,  # [T_r_k+1, u_phys_k, T_m_k]
                            'activation': 'tanh',
                            'dropout_rate': 0.01}
        }
    elif model_type == 'PhysNet':
        model = PhysNet
        network_param = {
            'lr': 0.001,
            'batch_size': 2048,
            'lambda_value': lambda_2,
            'encoding_network': {'input_size': 2 * (depth + 0),
                                 'fc': [24] * 1,
                                 'output_size': 1,  # [T_m]
                                 'activation': 'tanh',
                                 'dropout_rate': 0.01},
            'mdp_network': {'input_size': 5,  # [Time, T_r, T_m, u_k, T_a_k]
                            'fc': [128] * 1,
                            'output_size': 2,  # [T_r_k+1, u_phys_k]
                            'activation': 'tanh',
                            'dropout_rate': 0.05}
        }
    else:
        raise TypeError("Incorrect Model Type")
    # ---------------------------------------------------------------------------------------------------------------- #
    # Load Training data
    training_data_df = pd.read_csv(f'.\\data\\Training_data.csv')
    training_data_main = training_data_df.to_numpy()

    # Select required fields
    training_data_index_selection = tuple(
        [0, 1, *list(np.arange(4, 4 + depth)), *list(np.arange(4 + 24, 4 + 24 + depth)), 2,
         -1])  # [time, x_k, prev_state, previous_action, u_k, Ta_k]

    input_data_dict = {'x_agg_k': training_data_main[:, training_data_index_selection],
                       # [time, x_k, prev_state, u_k, Ta_k],
                       'label_k': training_data_main[:, (3, -2)]}  # [x_k1, u_phy_k]

    # Prepare data for training
    training_data_dict = prepare_data(input_data_dict)
    # ---------------------------------------------------------------------------------------------------------------- #
    # Seed and Initialize model
    torch.manual_seed(seed=model_seed)
    model_instance = model(network_param)

    # Add data to model
    model_instance.add_training_data(training_data_dict)

    # Initialize trainer and fit model
    trainer = pl.Trainer(max_epochs=75, min_epochs=1, gpus=0, progress_bar_refresh_rate=1,
                         weights_summary=None, checkpoint_callback=False, logger=False,
                         )
    trainer.fit(model_instance)
    # ---------------------------------------------------------------------------------------------------------------- #

