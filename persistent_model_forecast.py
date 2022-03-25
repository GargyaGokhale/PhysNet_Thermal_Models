"""
Validation script for trained models
21-09-2021
GG

"""
import os
import pickle
import numpy as np

if __name__ == '__main__':
    # ---------------------------------------------------------------------------------------------------------------- #
    working_dir = os.getcwd()
    parent_path = os.path.dirname(working_dir)

    # ---------------------------------------------------------------------------------------------------------------- #
    # Simulation Parameters
    training_data_type = 'simulated'

    expt_num = 1
    # ---------------------------------------------------------------------------------------------------------------- #

    # ---------------------------------------------------------------------------------------------------------------- #
    # File Handling
    result_path = os.path.join(parent_path, f'results')

    # Validation results file handling
    validation_main_dir = result_path
    # ---------------------------------------------------------------------------------------------------------------- #

    current_case = 'scenario_data\\Simulated_Data'
    scenario_path = os.path.join(parent_path, current_case)

    # ---------------------------------------------------------------------------------------------------------------- #

    # ------------------------------------------------------------------------------------------------------------ #
    # Test Data with different random action sequence
    temp_track = []
    building_mass_temp_track = []
    time_track = []
    plotting_time_track = []
    outside_temperature = []
    u_phys_track = []
    u_track = []

    # test_data_file = f'test_data_half_hour_size_{5}'
    test_data_file = f'test_data_half_hour_size_{5}_con'
    file_path = os.path.join(scenario_path, test_data_file)
    test_data_set = pickle.load(open(file_path, 'rb'))

    # print(f"Number of test days: {test_days}")
    persistent_model_room_temperature = []
    persistent_model_actual_action = []
    day_i = 0
    quarter_i = 0
    for index in range(test_data_set.shape[0]):
        temp_track.append(test_data_set[index, 1])
        time_track.append(test_data_set[index, 0])

        u_phys_track.append(test_data_set[index, -2])
        u_track.append(test_data_set[index, 2])
        outside_temperature.append(test_data_set[index, -1])
        plotting_time_track.append(day_i * 24 + (quarter_i * 30) / 60)
        
        if index == 0:
            persistent_model_room_temperature.append(test_data_set[index, 1])
            persistent_model_actual_action.append(test_data_set[index, -2])
        else:
            persistent_model_room_temperature.append(test_data_set[index-1, 1])
            persistent_model_actual_action.append(test_data_set[index-1, -2])

        quarter_i += 1
        if quarter_i % 48 == 0:
            day_i += 1
            quarter_i = 0
    # ------------------------------------------------------------------------------------------------------------ #
    persistent_model_error = np.mean(abs(np.array(persistent_model_room_temperature) - np.array(temp_track)))

    print(f"Persistent Model MAE = {persistent_model_error}")
