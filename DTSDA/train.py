import time
from DTSDA.alg.DTSDA import DGTSDA_temporal_diff,  DGTSDA_temporal_diff
from DTSDA.alg.opt import *
from DTSDA.alg import modelopera
from DTSDA.network.common_network import GradCAM, visualize_gradcam
from DTSDA.utils.util import set_random_seed, print_row, log_and_print


def DGTSDA_temporal_diff_train(S_torch_loader, T_torch_loader, ST_torch_loader, global_epoch, local_epoch, num_classes,
                               num_temporal_states,
                               conv1_in_channels, conv1_out_channels, conv2_out_channels,
                               kernel_size_num, in_features_size, bottleneck_dim, dis_hidden,
                               ReverseLayer_latent_domain_alpha,
                               ReverseLayer_domain_invariant_alpha, Entropylogits_lambda,
                               lr_decay1, lr_decay2, lr, optim_Adam_weight_decay, optim_Adam_beta, TICC_switch_penalty,
                               file_name):
    set_random_seed(1234)

    best_valid_acc, target_acc, best_cm = 0, 0, 0

    algorithm = DGTSDA_temporal_diff(num_classes, num_temporal_states, conv1_in_channels, conv1_out_channels,
                                     conv2_out_channels,
                                     kernel_size_num, in_features_size, bottleneck_dim, dis_hidden,
                                     ReverseLayer_latent_domain_alpha,
                                     ReverseLayer_domain_invariant_alpha, Entropylogits_lambda)

    algorithm.train()
    optd = get_optimizer(algorithm, lr_decay1, lr_decay2, lr, optim_Adam_weight_decay, optim_Adam_beta,
                         nettype='Diversify-adv')
    opt = get_optimizer(algorithm, lr_decay1, lr_decay2, lr, optim_Adam_weight_decay, optim_Adam_beta,
                        nettype='Diversify-cls')
    opta = get_optimizer(algorithm, lr_decay1, lr_decay2, lr, optim_Adam_weight_decay, optim_Adam_beta,
                         nettype='Diversify-all')

    grad_cam = GradCAM(algorithm.featurizer, target_layer_name="conv2")

    for round in range(global_epoch):
        log_and_print(content='\n========ROUND {' + str(round) + '}========', filename=file_name)
        log_and_print(content='====Feature update====', filename=file_name)
        print(f'\n========ROUND {round}========')
        print('====Feature update====')
        loss_list = ['class', 'domain']
        print_row(['epoch'] + [item + '_loss' for item in loss_list], colwidth=15, file_name=file_name)

        for step in range(local_epoch):
            for data in ST_torch_loader:
                loss_result_dict = algorithm.update_a(data, opta)
            print_row([step] + [loss_result_dict[item] for item in loss_list], colwidth=15, file_name=file_name)

        log_and_print(content='====Latent domain characterization====', filename=file_name)
        print('====Latent domain characterization====')
        loss_list = ['total', 'dis_class', 'dis_domain', 'ent_temporal_state']
        print_row(['epoch'] + [item + '_loss' for item in loss_list], colwidth=15, file_name=file_name)

        for step in range(local_epoch):
            for data in ST_torch_loader:
                loss_result_dict = algorithm.update_d(data, optd)
            print_row([step] + [loss_result_dict[item] for item in loss_list], colwidth=15, file_name=file_name)

        algorithm.set_dlabel_Temporal_Consistency(ST_torch_loader, S_torch_loader, T_torch_loader, TICC_switch_penalty)

        log_and_print(content='====Domain-invariant feature learning====', filename=file_name)
        print('====Domain-invariant feature learning====')
        loss_list = ['total', 'S_class', 'domain', 'ts']
        eval_dict = {'train': ['train_in'], 'valid': ['valid_in'], 'target': ['target_out'],
                     'target_cluster': ['target_out']}
        print_key = ['epoch']
        print_key.extend([item + '_loss' for item in loss_list])
        print_key.extend([item + '_acc' for item in eval_dict.keys()])
        print_key.append('total_cost_time')
        print_row(print_key, colwidth=15, file_name=file_name)

        sss = time.time()
        for step in range(local_epoch):
            for ST_data in ST_torch_loader:
                for S_data in S_torch_loader:
                    step_vals = algorithm.update(ST_data, S_data, opt)

            results = {'epoch': step, }

            results['train_acc'] = modelopera.accuracy(
                algorithm, S_torch_loader, None)

            # log_and_print(content='acc____________________________', filename=file_name)
            # print('acc____________________________')
            acc, S_mu, S_y = modelopera.accuracy(algorithm, S_torch_loader, None)
            results['valid_acc'] = acc

            # print('target_acc_#################################################################################')
            acc, cluster_acc, cm, T_mu, T_y = modelopera.accuracy_target_user(algorithm, T_torch_loader, S_torch_loader, None)
            results['target_acc'] = acc
            results['target_cluster_acc'] = cluster_acc
            results['cm'] = cm

            for key in loss_list:
                results[key + '_loss'] = step_vals[key]
            if results['target_cluster_acc'] > best_valid_acc:
                best_valid_acc = results['target_cluster_acc']
                target_acc = results['target_cluster_acc']
                best_cm = results['cm']
            results['total_cost_time'] = time.time() - sss
            print_row([results[key] for key in print_key], colwidth=23, file_name=file_name)

        #draw_TSNE(S_mu, S_y, T_mu, T_y, round)
        for data in T_torch_loader:
            x = data[0].float()
            y = data[1].long()
            cam = grad_cam.generate_heatmap()
            visualize_gradcam(cam, x.cpu().detach().numpy(), round, y.cpu().detach().numpy())

    print(f'Target acc: {target_acc:.4f}')
    print(best_cm)
    return target_acc, best_cm
