"""
Script to convert simulateed environment training data into csv

Date: 23-03-2022

-GG

"""

import os
import pickle
import numpy as np
import pandas as pd

if __name__ == '__main__':

    training_data_size = 120
    depth = 24

    working_dir = os.getcwd()
    parent_path = os.path.dirname(working_dir)
    result_path = os.path.join(parent_path, f'results')

    # ---------------------------------------------------------------------------------------------------------------- #
    current_case = 'scenario_data\\Simulated_Data'
    scenario_path = os.path.join(parent_path, current_case)

    training_data_file = f'training_data_half_hour_size_{training_data_size}'
    file_path = os.path.join(scenario_path, training_data_file)

    training_data_main = pickle.load(open(file_path, 'rb'))

    test_data_file = f'test_data_half_hour_size_{5}_con'
    file_path = os.path.join(scenario_path, test_data_file)
    test_data_set = pickle.load(open(file_path, 'rb'))

    test_hidden_temp_data_file = f'test_data_hidden_temp_half_hour_size_{5}_con'
    file_path = os.path.join(scenario_path, test_hidden_temp_data_file)
    hidden_temp_data_set = pickle.load(open(file_path, 'rb'))

    # Convert Training, Test, Hidden Temp data into csv
    hidden_temp_array = np.zeros([len(hidden_temp_data_set), 2])

    for i in range(len(hidden_temp_data_set)):
        hidden_temp_array[i, 0] = test_data_set[i, 0]           # Time
        hidden_temp_array[i, 1] = hidden_temp_data_set[i, 0]

    column_list = ['Time', 'Current_State', 'Current_Action', 'Next_State']

    for d in range(depth):
        column_list.append(f"Previous_State_t-{(d+1)}")

    for d in range(depth):
        column_list.append(f"Previous_u_phys_t-{(d+1)}")

    column_list.append(f"Current_u_phys")
    column_list.append(f"Outside_Air_Temperature")

    hidden_temp_column_list = ["Time", "Hidden_State"]

    training_df = pd.DataFrame(training_data_main, columns=column_list)
    test_df = pd.DataFrame(test_data_set, columns=column_list)
    hidden_temp_df = pd.DataFrame(hidden_temp_array, columns=hidden_temp_column_list)

    training_df.to_csv(path_or_buf=os.path.join(scenario_path, "Training_data.csv"), index=False)
    test_df.to_csv(path_or_buf=os.path.join(scenario_path, "Test_data.csv"), index=False)
    hidden_temp_df.to_csv(path_or_buf=os.path.join(scenario_path, "Hidden_Temperature_data.csv"), index=False)
    print(f"Done")

