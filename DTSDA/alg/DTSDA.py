from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
from DGTSDA.network import Adver_network, common_network, act_network
from DGTSDA.loss.common_loss import Entropylogits
from TICC.TICC_utils import updateClusters
import torch.utils.data as Data


class DGTSDA_temporal_diff(torch.nn.Module):

    def __init__(self, num_classes, num_temporal_states, conv1_in_channels, conv1_out_channels, conv2_out_channels,
                 kernel_size_num, in_features_size, bottleneck_dim, dis_hidden, ReverseLayer_latent_domain_alpha,
                 ReverseLayer_domain_invariant_alpha, Entropylogits_lambda):

        super(DGTSDA_temporal_diff, self).__init__()
        self.num_classes = num_classes
        self.num_temporal_states = num_temporal_states

        self.conv1_in_channels = conv1_in_channels
        self.conv1_out_channels = conv1_out_channels
        self.conv2_out_channels = conv2_out_channels
        self.kernel_size_num = kernel_size_num

        self.in_features_size = in_features_size
        self.bottleneck_dim = bottleneck_dim
        self.dis_hidden = dis_hidden

        self.ReverseLayer_latent_domain_alpha = ReverseLayer_latent_domain_alpha
        self.ReverseLayer_domain_invariant_alpha = ReverseLayer_domain_invariant_alpha
        self.Entropylogits_lambda = Entropylogits_lambda

        self.featurizer = act_network.ActNetwork(self.conv1_in_channels, self.conv1_out_channels,
                                                 self.conv2_out_channels, self.kernel_size_num, self.in_features_size)

        self.abottleneck = common_network.feat_bottleneck(self.in_features_size, self.bottleneck_dim)
        self.aclassifier = common_network.feat_classifier(class_num=2 * self.num_classes * self.num_temporal_states,
                                                          bottleneck_dim=self.bottleneck_dim)
        self.adomain_classifier = common_network.feat_classifier(class_num=2, bottleneck_dim=self.bottleneck_dim)

        self.dbottleneck = common_network.feat_bottleneck(self.in_features_size, self.bottleneck_dim)
        self.ddiscriminator = Adver_network.Discriminator(self.bottleneck_dim, self.dis_hidden, 2 * self.num_classes)
        self.ddomain_discriminator = Adver_network.Discriminator(self.bottleneck_dim, self.dis_hidden, 2)
        self.dclassifier = common_network.feat_classifier(self.num_temporal_states, self.bottleneck_dim)

        self.bottleneck = common_network.feat_bottleneck(self.in_features_size, self.bottleneck_dim)
        self.classifier = common_network.feat_classifier(class_num=self.num_classes,
                                                         bottleneck_dim=self.bottleneck_dim)
        self.discriminator = Adver_network.Discriminator(self.bottleneck_dim, self.dis_hidden, 2)
        self.classifier_ts = common_network.feat_classifier(class_num=self.num_temporal_states,
                                                            bottleneck_dim=self.bottleneck_dim)

    def update_d(self, minibatch, opt):
        all_x1 = minibatch[0].float()
        all_c1 = minibatch[1].long()
        all_ts1 = minibatch[2].long()
        all_d1 = minibatch[3].long()
        z1 = self.dbottleneck(self.featurizer(all_x1))

        ts1 = self.dclassifier(z1)
        ent_loss = self.Entropylogits_lambda * Entropylogits(ts1) + F.cross_entropy(ts1, all_ts1)

        disc_class_in1 = Adver_network.ReverseLayerF.apply(z1, self.ReverseLayer_latent_domain_alpha)
        disc_class_out1 = self.ddiscriminator(disc_class_in1)
        disc_class_loss = F.cross_entropy(disc_class_out1, all_c1, reduction='mean')

        disc_d_in1 = Adver_network.ReverseLayerF.apply(z1, self.ReverseLayer_latent_domain_alpha)
        disc_d_out1 = self.ddomain_discriminator(disc_d_in1)
        disc_d_loss = F.cross_entropy(disc_d_out1, all_d1, reduction='mean')

        loss = ent_loss + disc_class_loss + disc_d_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {'total': loss.item(), 'dis_class': disc_class_loss.item(), 'dis_domain': disc_d_loss.item(),
                'ent_temporal_state': ent_loss.item()}

    def set_dlabel_Temporal_Consistency(self, ST_torch_loader, S_torch_loader, T_torch_loader, TICC_switch_penalty):
        self.dbottleneck.eval()
        self.dclassifier.eval()
        self.featurizer.eval()

        start_test = True
        with torch.no_grad():
            iter_test = iter(ST_torch_loader)
            for _ in range(len(ST_torch_loader)):
                data = next(iter_test)
                inputs = data[0]
                inputs = inputs.float()
                index = data[-1]
                feas = self.dbottleneck(self.featurizer(inputs))
                outputs = self.dclassifier(feas)
                if start_test:
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_index = index
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat(
                        (all_output, outputs.float().cpu()), 0)
                    all_index = np.hstack((all_index, index))
        all_output = nn.Softmax(dim=1)(all_output)

        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()

        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])  # inital central points
        dd = cdist(all_fea, initc, 'cosine')
        class_list = ST_torch_loader.dataset.tensors[1].numpy()
        unique_elements = np.unique(class_list)
        idx_list = ST_torch_loader.dataset.tensors[4].numpy()
        pred_label = np.empty((0, ))
        for a_class in range(len(unique_elements)):
            this_class_idx_list = []
            for index, class_name in enumerate(class_list):
                if class_name == a_class:
                    this_class_idx_list.append(idx_list[index])
            this_class_LLE_all_points_clusters = dd[this_class_idx_list]
            this_class_pred_label = updateClusters(this_class_LLE_all_points_clusters, switch_penalty=TICC_switch_penalty)
            pred_label = np.concatenate((pred_label, this_class_pred_label), axis=0)
        pred_label = pred_label.astype(int)

        for _ in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_fea, initc, 'cosine')
            pred_label = dd.argmin(axis=1)

        ST_torch_loader.dataset.tensors = (
            ST_torch_loader.dataset.tensors[0], ST_torch_loader.dataset.tensors[1], torch.tensor(pred_label),
            ST_torch_loader.dataset.tensors[3], ST_torch_loader.dataset.tensors[4])

        S_torch_loader.dataset.tensors = (
            S_torch_loader.dataset.tensors[0], S_torch_loader.dataset.tensors[1],
            torch.tensor(pred_label[0:len(S_torch_loader.dataset.tensors[0])]),
            S_torch_loader.dataset.tensors[3], S_torch_loader.dataset.tensors[4])

        T_torch_loader.dataset.tensors = (
            T_torch_loader.dataset.tensors[0], T_torch_loader.dataset.tensors[1],
            torch.tensor(pred_label[
                         len(S_torch_loader.dataset.tensors[0]): len(S_torch_loader.dataset.tensors[0]) + len(
                             T_torch_loader.dataset.tensors[0])]),
            T_torch_loader.dataset.tensors[3], T_torch_loader.dataset.tensors[4])

        #print(Counter(pred_label))
        self.dbottleneck.train()
        self.dclassifier.train()
        self.featurizer.train()

    def set_dlabel(self, ST_torch_loader, S_torch_loader, T_torch_loader):
        self.dbottleneck.eval()
        self.dclassifier.eval()
        self.featurizer.eval()

        start_test = True
        with torch.no_grad():
            iter_test = iter(ST_torch_loader)
            for _ in range(len(ST_torch_loader)):
                data = next(iter_test)
                inputs = data[0]
                inputs = inputs.float()
                index = data[-1]
                feas = self.dbottleneck(self.featurizer(inputs))
                outputs = self.dclassifier(feas)
                if start_test:
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_index = index
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat(
                        (all_output, outputs.float().cpu()), 0)
                    all_index = np.hstack((all_index, index))
        all_output = nn.Softmax(dim=1)(all_output)

        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()

        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)

        for _ in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_fea, initc, 'cosine')
            pred_label = dd.argmin(axis=1)

        ST_torch_loader.dataset.tensors = (
            ST_torch_loader.dataset.tensors[0], ST_torch_loader.dataset.tensors[1], torch.tensor(pred_label),
            ST_torch_loader.dataset.tensors[3], ST_torch_loader.dataset.tensors[4])

        S_torch_loader.dataset.tensors = (
            S_torch_loader.dataset.tensors[0], S_torch_loader.dataset.tensors[1],
            torch.tensor(pred_label[0:len(S_torch_loader.dataset.tensors[0])]),
            S_torch_loader.dataset.tensors[3], S_torch_loader.dataset.tensors[4])

        T_torch_loader.dataset.tensors = (
            T_torch_loader.dataset.tensors[0], T_torch_loader.dataset.tensors[1],
            torch.tensor(pred_label[
                         len(S_torch_loader.dataset.tensors[0]): len(S_torch_loader.dataset.tensors[0]) + len(
                             T_torch_loader.dataset.tensors[0])]),
            T_torch_loader.dataset.tensors[3], T_torch_loader.dataset.tensors[4])

        #print(Counter(pred_label))
        self.dbottleneck.train()
        self.dclassifier.train()
        self.featurizer.train()

    def update(self, ST_data, S_data, opt):
        ST_all_x = ST_data[0].float()
        ST_all_ts = ST_data[2].long()
        ST_all_d = ST_data[3].long()
        ST_all_z = self.bottleneck(self.featurizer(ST_all_x))

        disc_d_input = Adver_network.ReverseLayerF.apply(ST_all_z, self.ReverseLayer_domain_invariant_alpha)
        disc_d_out = self.discriminator(disc_d_input)
        disc_d_loss = F.cross_entropy(disc_d_out, ST_all_d)

        ST_all_ts_preds = self.classifier_ts(ST_all_z)
        ST_classifier_ts_loss = F.cross_entropy(ST_all_ts_preds, ST_all_ts)

        S_all_x = S_data[0].float()
        S_all_c = S_data[1].long()
        S_all_ts = S_data[2].long()
        S_all_z = self.bottleneck(self.featurizer(S_all_x))
        S_all_preds = self.classifier(S_all_z)
        S_classifier_loss = F.cross_entropy(S_all_preds, S_all_c)

        loss = S_classifier_loss + disc_d_loss + ST_classifier_ts_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {'total': loss.item(), 'S_class': S_classifier_loss.item(), 'domain': disc_d_loss.item(),
                'ts': ST_classifier_ts_loss.item()}

    def update_a(self, minibatches, opt):
        all_x = minibatches[0].float()
        all_c = minibatches[1].long()
        all_ts = minibatches[2].long()
        all_d = minibatches[3].long()

        all_y = all_ts * 2 * self.num_classes + all_c
        all_z = self.abottleneck(self.featurizer(all_x))
        all_preds = self.aclassifier(all_z)
        classifier_loss = F.cross_entropy(all_preds, all_y)

        all_d_preds = self.adomain_classifier(all_z)
        d_classifier_loss = F.cross_entropy(all_d_preds, all_d)

        loss = classifier_loss + d_classifier_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {'class': classifier_loss.item(), 'domain': d_classifier_loss.item()}

    def predict(self, x):
        return self.classifier(self.bottleneck(self.featurizer(x))), self.bottleneck(self.featurizer(x))

    def predict1(self, x):
        return self.ddiscriminator(self.dbottleneck(self.featurizer(x)))
