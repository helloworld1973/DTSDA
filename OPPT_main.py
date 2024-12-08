import torch
import numpy as np
from gtda.time_series import SlidingWindow
import random

from DTSDA.train import DGTSDA_temporal_diff_train
from utils import get_DGTSDA_temporal_diff_train_data

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# OPPT_dataset
sensor_channels_required = ['IMU_RLA_ACC_X', 'IMU_RLA_ACC_Y', 'IMU_RLA_ACC_Z',
                            'IMU_RLA_GYRO_X', 'IMU_RLA_GYRO_Y', 'IMU_RLA_GYRO_Z']  # right lower arm
activity_list = ['Stand', 'Walk', 'Sit', 'Lie']
DATASET_NAME = 'OPPT'
activities_required = activity_list
source_user = 'S1'
target_user = 'S2'  # S3

Sampling_frequency = 30  # HZ
Num_Seconds = 3
Window_Overlap_Rate = 0.5


def sliding_window_seg(data_x, data_y):
    # same setting as M1, except for no feature extraction step
    sliding_bag = SlidingWindow(size=int(Sampling_frequency * Num_Seconds),
                                stride=int(Sampling_frequency * Num_Seconds * (1 - Window_Overlap_Rate)))
    X_bags = sliding_bag.fit_transform(data_x)
    Y_bags = sliding_bag.resample(data_y)  # last occur label
    Y_bags = Y_bags.tolist()

    return X_bags, Y_bags


S_data = []
S_label = []
T_data = []
T_label = []

for index, a_act in enumerate(activities_required):
    with open('./gen_data/' + DATASET_NAME + '_' + a_act + '_' + str(source_user) + '_X_features.npy', 'rb') as f:
        source_bags = np.load(f, allow_pickle=True)
    with open('./gen_data/' + DATASET_NAME + '_' + a_act + '_' + str(source_user) + '_Y_labels.npy', 'rb') as f:
        source_labels = np.load(f)
    with open('./gen_data/' + DATASET_NAME + '_' + a_act + '_' + str(target_user) + '_X_features.npy', 'rb') as f:
        target_bags = np.load(f, allow_pickle=True)
    with open('./gen_data/' + DATASET_NAME + '_' + a_act + '_' + str(target_user) + '_Y_labels.npy', 'rb') as f:
        target_labels = np.load(f)

    s_X_bags, s_Y_bags = sliding_window_seg(source_bags, source_labels)
    t_X_bags, t_Y_bags = sliding_window_seg(target_bags, target_labels)

    if index == 0:
        S_data = s_X_bags
        S_label = s_Y_bags
        T_data = t_X_bags
        T_label = t_Y_bags
    else:
        S_data = np.vstack((S_data, s_X_bags))
        S_label = S_label + s_Y_bags
        T_data = np.vstack((T_data, t_X_bags))
        T_label = T_label + t_Y_bags
print()
S_label = [int(x) for x in S_label]
T_label = [int(x) for x in T_label]
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# model training paras settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 1e-2
num_D = 6
width = int(Sampling_frequency * Num_Seconds)
Num_classes = 4
Epochs = 200
Local_epoch = 1
cuda = False
print()
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# DTSDA_temporal_diff model
Num_temporal_states = 5  # 2 3 4 5 6
Conv1_in_channels = num_D
Conv1_out_channels = 16
Conv2_out_channels = 32
Kernel_size_num = 9
In_features_size = Conv2_out_channels * 16
Bottleneck_dim = 100
Dis_hidden = 50
ReverseLayer_latent_domain_alpha = 0.1
ReverseLayer_domain_invariant_alpha = 0.1
Entropylogits_lambda = 0.0
Lr_decay1 = 1.0  # 0.5 1.0
Lr_decay2 = 1.0  # 0.5 1.0
Optim_Adam_weight_decay = 5e-4  # 5e-4
Optim_Adam_beta = 0.2  # 0.5
TICC_switch_penalty = 0.0001  # 0.001  0.01
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
file_name = 'M2_new_DTSDA_' + str(DATASET_NAME) + '_' + str(source_user) + '_' + str(target_user) + '_output.txt'
file_name_summary = 'M2_new_DTSDA_' + str(DATASET_NAME) + '_' + str(source_user) + '_' + str(target_user) + '_output_summary.txt'

for Bottleneck_dim in [100, 50]:
    for Dis_hidden in [50, 20]:
        for Lr_decay1 in [1.0, 0.5]:  # 0.5 1.0
            for Lr_decay2 in [1.0, 0.5]:
                for Optim_Adam_weight_decay in [5e-4, 5e-2]:
                    for Optim_Adam_beta in [0.2, 0.5]:
                        for lr in [1e-2, 1e-1, 1e-3]:
                            for Num_temporal_states in [2, 3, 4, 5, 6, 7]:
                                for TICC_switch_penalty in [0.01,  0.1, 1]: # 0, 0.0001, 0.001,
                                    print('para_setting:' + str(Num_temporal_states) + '_' + str(
                                        Bottleneck_dim) + '_' + str(Dis_hidden) + '_' + str(
                                        Lr_decay1) + '_' + str(Lr_decay2) + '_' + str(
                                        Optim_Adam_weight_decay) + '_' + str(Optim_Adam_beta) + '_' + str(
                                        TICC_switch_penalty))

                                    S_torch_loader, T_torch_loader, ST_torch_loader = get_DGTSDA_temporal_diff_train_data(
                                        S_data, S_label, T_data, T_label,
                                        batch_size=10000, num_D=num_D,
                                        width=width,
                                        num_class=Num_classes)
                                    DGTSDA_temporal_diff_train(S_torch_loader, T_torch_loader, ST_torch_loader,
                                                               global_epoch=Epochs,
                                                               local_epoch=Local_epoch,
                                                               num_classes=Num_classes,
                                                               num_temporal_states=Num_temporal_states,
                                                               conv1_in_channels=Conv1_in_channels,
                                                               conv1_out_channels=Conv1_out_channels,
                                                               conv2_out_channels=Conv2_out_channels,
                                                               kernel_size_num=Kernel_size_num,
                                                               in_features_size=In_features_size,
                                                               bottleneck_dim=Bottleneck_dim, dis_hidden=Dis_hidden,
                                                               ReverseLayer_latent_domain_alpha=ReverseLayer_latent_domain_alpha,
                                                               ReverseLayer_domain_invariant_alpha=ReverseLayer_domain_invariant_alpha,
                                                               Entropylogits_lambda=Entropylogits_lambda,
                                                               lr_decay1=Lr_decay1, lr_decay2=Lr_decay2, lr=lr,
                                                               optim_Adam_weight_decay=Optim_Adam_weight_decay,
                                                               optim_Adam_beta=Optim_Adam_beta,
                                                               TICC_switch_penalty=TICC_switch_penalty,
                                                               file_name = file_name)

                                    print()
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

