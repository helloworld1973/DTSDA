import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import softmax
import shap
from shap.explainers import DeepExplainer
from torch.autograd import Variable

from DTSDA.network.common_network import GradCAM


def accuracy_target_user(network, loader, s_loader, weights, usedpredict='p'):
    correct = 0
    total = 0
    weights_offset = 0
    total_cluster_accuracy = 0
    s_length = len(s_loader.dataset.tensors[1])

    confusion_matrix_all = 0

    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].float()
            y = data[1].long()
            if usedpredict == 'p':
                p, mu = network.predict(x)
                '''
                # Example input tensor (replace with your actual tensor)
                #x = torch.randn(1557, 6, 1, 300)  # Example shape: (samples, channels, height, width)
                x_numpy = x.cpu().detach().numpy()  # Convert to NumPy

                # Reshape input to 2D
                x_flat = x_numpy.reshape(x_numpy.shape[0], -1)  # Shape: (1557, 6*1*300)

                # Summarize background data (use shap.sample or shap.kmeans)
                background_data = shap.kmeans(x_flat, 10)  # Cluster into 10 groups

                # Define a callable prediction function for SHAP
                def predict_fn(data):
                    data_tensor = torch.tensor(data).reshape(-1, 6, 1, 300)  # Reshape back for the model
                    return network.predict_shap(data_tensor).cpu().detach().numpy()

                # Initialize KernelExplainer
                explainer = shap.KernelExplainer(predict_fn, background_data)

                # Compute SHAP values
                shap_values = explainer.shap_values(x_flat)

                # Visualize SHAP summary plot
                shap.summary_plot(shap_values, x_flat)
                '''

            else:
                p = network.predict1(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset:
                                        weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() *
                            batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() *
                            batch_weights).sum().item()

                # if cluster in each class first, then majority voting for one label in each cluster in each class
                class_list = y.numpy()
                class_unique_elements = np.unique(class_list)
                ts_list = data[2].numpy().astype(int)
                ts_unique_elements = np.unique(ts_list)
                idx_list = loader.dataset.tensors[4].numpy()

                # Step 1: Initialize the confusion matrix
                confusion_matrix = np.zeros((class_unique_elements.shape[0], class_unique_elements.shape[0]))

                for a_class in range(len(class_unique_elements)):
                    for a_ts_label in range(len(ts_unique_elements)):
                        this_class_this_ts_idx_list = []
                        for index, class_name in enumerate(class_list):
                            if class_name == a_class and ts_list[index] == a_ts_label:
                                this_class_this_ts_idx_list.append(idx_list[index] - s_length)

                        if len(this_class_this_ts_idx_list) == 0:
                            aaaaa=1
                            #print(str(a_class) + '_' + str(a_ts_label) + ': None list')
                        else:
                            this_class_this_ts_p = p[this_class_this_ts_idx_list]
                            this_class_this_ts_y = y[this_class_this_ts_idx_list]
                            this_class_this_ts_num = len(this_class_this_ts_idx_list)
                            this_class_this_ts_correct = (this_class_this_ts_p.argmax(1).eq(
                                this_class_this_ts_y).float()).sum().item() / this_class_this_ts_num

                            # calculate cluster accuracy
                            average_dis_per_class = torch.mean(this_class_this_ts_p, dim=0,
                                                               keepdim=True)
                            average_dis_per_class = torch.softmax(average_dis_per_class, dim=1)

                            # update confusion matrix
                            max_class = average_dis_per_class.argmax(1)
                            confusion_matrix[a_class][max_class.item()] += this_class_this_ts_num


                            total_cluster_accuracy += this_class_this_ts_num * (max_class.eq(a_class).float().numpy()[0])

                            #print(str(a_class) + '_' + str(a_ts_label) + ':' + str(this_class_this_ts_correct) + '___:' + str(average_dis_per_class) + '___:' + str(this_class_this_ts_num))

                confusion_matrix_all = confusion_matrix

            total += batch_weights.sum().item()
    network.train()

    return correct / total, total_cluster_accuracy / total, confusion_matrix_all, mu, y


def accuracy(network, loader, weights, usedpredict='p'):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].float()
            y = data[1].long()
            if usedpredict == 'p':
                p, mu = network.predict(x)
            else:
                p = network.predict1(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset:
                                        weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() *
                            batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() *
                            batch_weights).sum().item()

            total += batch_weights.sum().item()
    network.train()

    return correct / total, mu, y





