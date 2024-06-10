import torch


def get_params(alg, lr_decay1, lr_decay2, init_lr, nettype):
    if nettype == 'Diversify-adv':
        params = [
            {'params': alg.dbottleneck.parameters(), 'lr': lr_decay2 * init_lr},
            {'params': alg.dclassifier.parameters(), 'lr': lr_decay2 * init_lr},
            {'params': alg.ddiscriminator.parameters(), 'lr': lr_decay2 * init_lr},
            {'params': alg.ddomain_discriminator.parameters(), 'lr': lr_decay2 * init_lr}

        ]
        return params
    elif nettype == 'Diversify-cls':
        params = [
            {'params': alg.bottleneck.parameters(), 'lr': lr_decay2 * init_lr},
            {'params': alg.classifier.parameters(), 'lr': lr_decay2 * init_lr},
            {'params': alg.discriminator.parameters(), 'lr': lr_decay2 * init_lr}
        ]
        return params
    elif nettype == 'Diversify-all':
        params = [
            {'params': alg.featurizer.parameters(), 'lr': lr_decay1 * init_lr},
            {'params': alg.abottleneck.parameters(), 'lr': lr_decay2 * init_lr},
            {'params': alg.aclassifier.parameters(), 'lr': lr_decay2 * init_lr},
            {'params': alg.adomain_classifier.parameters(), 'lr': lr_decay2 * init_lr}
        ]
        return params


def get_optimizer(alg, lr_decay1, lr_decay2, lr, optim_Adam_weight_decay, optim_Adam_beta, nettype):
    params = get_params(alg, lr_decay1, lr_decay2, lr, nettype=nettype)
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=optim_Adam_weight_decay, betas=(optim_Adam_beta, 0.9))
    return optimizer
