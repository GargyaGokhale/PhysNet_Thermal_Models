"""
Class defining PhysNet Variant

Date: 23-03-2021

Author: Gargya Gokhale

"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from utils.support_functions import transform_temp, inverse_transform_temp, inverse_transform_action, \
    inverse_transform_outside_temp
from utils.support_functions import fc_module


class PhysNet(pl.LightningModule):

    def __init__(self, parameter_dict=None):
        super().__init__()

        if parameter_dict is None:
            parameter_dict = {
                'lr': 0.005, 'batch_size': 128, 'lambda_value': 2.0,
                'encoding_network': {'input_size': 2,  # [x_t-1, x_t-2]
                                     'fc': [16, ],
                                     'output_size': 1,  # [T_m]
                                     'activation': 'tanh',
                                     'dropout_rate': 0.0},
                'mdp_network': {'input_size': 5,  # [Time, T_r, T_m, u_k, T_a_k]
                                'fc': [16, ],
                                'output_size': 2,  # [T_r_k+1, u_phys_k]
                                'activation': 'tanh',
                                'dropout_rate': 0.0}

            }

        self.parameter_dict = parameter_dict
        self.training_data_size = None

        self.encoding_network_params = parameter_dict['encoding_network']
        self.mdp_network_params = parameter_dict['mdp_network']

        self.encoding_network = nn.Sequential(*self.make_network(network_params=self.encoding_network_params))
        # self.encoding_network.apply(xavier_weight_initialisation)

        self.mdp_network = nn.Sequential(*self.make_network(network_params=self.mdp_network_params))
        # self.mdp_network.apply(xavier_weight_initialisation)

        # Physics Parameters

        self.c11 = nn.Parameter(torch.tensor([4e-04]))
        self.c12 = nn.Parameter(torch.tensor([3.33e-04]))
        self.c21 = nn.Parameter(torch.tensor([2.0e-05]))
        self.c22 = nn.Parameter(torch.tensor([2.0e-05]))
        self.b1 = nn.Parameter(torch.tensor([2.50e-07]))
        self.d11 = nn.Parameter(torch.tensor([4e-08]))
        self.d12 = nn.Parameter(torch.tensor([4e-08]))
        self.d13 = nn.Parameter(torch.tensor([6.66e-05]))
        self.d21 = nn.Parameter(torch.tensor([2.5e-09]))
        self.d22 = nn.Parameter(torch.tensor([2.5e-09]))
        self.d23 = nn.Parameter(torch.tensor([0.0]))

        # Model Parameters
        self.lr = parameter_dict['lr']
        self.batch_size = parameter_dict['batch_size']

        self.loss = None
        self.training_loss = {'Prediction Loss': [],
                              'Model Loss': [],
                              'Constraint Loss': [],
                              'Total Loss': []}

        self.x_agg_k_data = None
        self.label_k_data = None
        self.x_agg_k1_data = None
        self.label_k1_data = None

        self.lamda_1 = 1
        self.lamda_2 = parameter_dict['lambda_value']


    @staticmethod
    def make_network(network_params):
        """
        :type network_params: dict with keys: input_size, fc, output_size, activation, dropout_rate
        """
        if len(network_params['fc']) == 0:
            network = [fc_module([network_params['input_size'], network_params['output_size']],
                                 activation=network_params['activation'], dropout_rate=network_params['dropout_rate'])]
        else:
            network = [fc_module([network_params['input_size'], network_params['fc'][0]],
                                 activation=network_params['activation'], dropout_rate=network_params['dropout_rate'])]
            for l_i in range(len(network_params['fc'][:-1])):
                network += [fc_module([network_params['fc'][l_i], network_params['fc'][l_i + 1]],
                                      activation=network_params['activation'],
                                      dropout_rate=network_params['dropout_rate'])]
            network += [fc_module([network_params['fc'][-1], network_params['output_size']],
                                  activation=network_params['activation'], dropout_rate=network_params['dropout_rate'])]
        return network

    def forward(self, x1):  # x1: [time, current_state, previous_states, action, outside_temp]
        x_state = x1[:, 2:-2]  # [previous_states]
        x_T = x1[:, (0, 1)]  # [Time, T_r]
        u_T_a = x1[:, (-2, -1)]  # [u_k, T_a_k]

        # Get x_M,k
        x_M_k = (self.encoding_network(x_state))  # xM,k = [T_m]

        # Get x_o,k+1
        x = torch.cat([x_T, x_M_k, u_T_a], dim=1)  # [Time, T_r, T_m, u, T_a]
        x_o_k1 = (self.mdp_network(x))  # [x_o_k1, u_phys_k]
        return x_o_k1, x_M_k

    @torch.no_grad()
    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        o_k_1, _ = self.forward(x)
        o_k_1 = (o_k_1.data.numpy())

        return o_k_1

    @torch.no_grad()
    def model_encoded_state(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        _, x_M_k = self.forward(x)

        return x_M_k.data.numpy()

    def configure_optimizers(self):
        # set different learning rate for model physics params
        physics_param_list = []
        physics_params = []
        base_params = []
        for name, param in self.named_parameters():
            if 'network' not in name:
                physics_param_list.append(str(name))
                physics_params.append(param)
            else:
                base_params.append(param)

        optimiser = optim.Adam(
            [{'params': physics_params, 'lr': 1e-7, 'weight_decay': 1e-10},
             {'params': base_params, 'lr': self.lr, 'weight_decay': 1e-5}
             ]
        )

        lr_scheduler = {'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5),
                        'monitor': 'loss'}

        return [optimiser], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        x_agg_k, o_k1, x_agg_k1, o_k2 = batch

        # x_agg_k = x_o_k    x_agg_k2 = x_o_k+1
        x_o_k1, x_M_k = self.forward(x_agg_k)
        x_o_k2, x_M_k1 = self.forward(x_agg_k1)

        # Rescale
        T_r_k_unscale = inverse_transform_temp(x_agg_k[:, 1])
        T_r_k1_unscale = inverse_transform_temp(x_o_k1[:, 0])

        T_r_k2_unscale = inverse_transform_temp(o_k2[:, 0])
        u_phys_k1_unscale = inverse_transform_action(o_k2[:, -1])
        T_a_k1_unscale = inverse_transform_outside_temp(x_agg_k1[:, -1])

        physics_block_actual_inputs = {
            'T_r_k': T_r_k_unscale.view(-1, 1),
            'T_r_k1': T_r_k1_unscale.view(-1, 1),
            'T_r_k2': T_r_k2_unscale.view(-1, 1),
            'u_phys_k1': u_phys_k1_unscale.view(-1, 1),
            'T_a_k1': T_a_k1_unscale.view(-1, 1)
        }

        T_m_k1_estimate = self.physics_bloc(actual_values=physics_block_actual_inputs)

        target_dict = {'x_k_o2': x_o_k2[:, 0],
                       'u_phys_k1': x_o_k2[:, 1],
                       'T_m_k': T_m_k1_estimate
                       }

        prediction_dict = {'x_k_o2': o_k2[:, 0],
                           'u_phys_k1': o_k2[:, 1],
                           'T_m_k': x_M_k1
                           }

        loss, prediction_loss, model_loss, constrain_loss = self.constrained_loss(prediction_dict, target_dict)

        loss_dict = {'loss': loss}
        self.loss = loss.data
        self.log("loss", self.loss)

        self.training_loss['Total Loss'].append(loss.data.numpy())
        self.training_loss['Prediction Loss'].append(prediction_loss.data.numpy())
        self.training_loss['Model Loss'].append(model_loss.data.numpy())
        self.training_loss['Constraint Loss'].append(constrain_loss.data.numpy())

        return loss_dict

    def physics_bloc(self, actual_values):

        T_r_k_actual = actual_values['T_r_k']
        T_r_k1_actual = actual_values['T_r_k1']
        T_r_k2_actual = actual_values['T_r_k2']
        u_phys_k1_actual = actual_values['u_phys_k1']
        T_a_k1_actual = actual_values['T_a_k1']

        delta_t = 30 * 60
        d_T_r_k1_actual = (((T_r_k1_actual - T_r_k_actual) / delta_t) + ((T_r_k2_actual - T_r_k1_actual) / delta_t))/2
        # d_T_r_k1_actual = ((T_r_k1_actual - T_r_k_actual) / delta_t)
        T_m_k1_estimate = (d_T_r_k1_actual + (self.c11 * T_r_k1_actual - self.b1 * u_phys_k1_actual - (self.c11 - self.c12) * T_a_k1_actual)) / (
                             self.c12)
        convolution_T_m_k1_estimate = (F.conv1d(T_m_k1_estimate.view(1, 1, -1), torch.ones(1, 1, 5) / 5, padding=(2))).view(-1, 1)
        T_m_k1_estimate_scaled = transform_temp(convolution_T_m_k1_estimate)

        return T_m_k1_estimate_scaled

    def constrained_loss(self, prediction_dict, target_dict):
        l12 = 1 * F.mse_loss(prediction_dict['x_k_o2'], target_dict['x_k_o2'])
        l22 = 1 * F.mse_loss(prediction_dict['u_phys_k1'], target_dict['u_phys_k1'])

        l3 = 1 * F.mse_loss(prediction_dict['T_m_k'], target_dict['T_m_k'])

        l41 = torch.relu(-self.c11)
        l42 = torch.relu(-self.c12)
        l43 = torch.relu(-self.c21)
        l44 = torch.relu(-self.b1)
        l45 = torch.relu(-self.d13)
        l51 = torch.relu((self.c11 - self.c12) * -1)  # c11 > c12
        l52 = torch.relu((self.c11 - 3.5*self.c21) * -1)
        # l8 = 0

        constrained_loss = 1e6 * (l41 + l42 + l43 + l44 + l45 + l51 + l52)

        prediction_loss = (l12 + l22)
        model_loss = l3

        loss = (self.lamda_1 * prediction_loss + self.lamda_2 * model_loss + self.lamda_2 * constrained_loss) * 1

        return loss, prediction_loss, model_loss, constrained_loss

    def add_training_data(self, main_data_dict):

        self.x_agg_k_data = (main_data_dict['x_agg_k'])
        self.label_k_data = (main_data_dict['label_k'])
        self.x_agg_k1_data = (main_data_dict['x_agg_k1'])
        self.label_k1_data = (main_data_dict['label_k1'])

    def train_dataloader(self):
        training_set = TensorDataset(torch.tensor(self.x_agg_k_data, dtype=torch.float32),
                                     torch.tensor(self.label_k_data, dtype=torch.float32),
                                     torch.tensor(self.x_agg_k1_data, dtype=torch.float32),
                                     torch.tensor(self.label_k1_data, dtype=torch.float32))
        training_data_loader = DataLoader(training_set, shuffle=False, batch_size=self.batch_size)
        return training_data_loader
