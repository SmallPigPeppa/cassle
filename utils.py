from models.resnet_modified import resnet18
def get_modified_state_dict(state_dict):
    import collections
    modified_state_dict = collections.OrderedDict()
    # modified_resnet=resnet18()
    # modified_resnet_keys=[i for i,_ in modified_resnet.named_parameters()]
    # for key, value in state_dict.items():
    #     if 'conv' in key and 'encoder' in key and 'layer' in key:
    #         a = key.split('.')
    #         key = f'{a[0]}.{a[1]}.{a[2]}.{a[3]}.conv2d_3x3.{a[4]}'
    #     modified_state_dict[key] = value
    # return modified_state_dict

    for key, value in state_dict.items():
        if 'conv' in key and 'encoder' in key and 'layer' in key:
            # print(key)
            a = key.split('.')
            key = f'{a[0]}.{a[1]}.{a[2]}.{a[3]}.conv2d_3x3.{a[4]}'
            # print(key2)
        modified_state_dict[key] = value
    return modified_state_dict





if __name__=='__main__':
    # import torch.nn as nn
    # a=nn.Conv2d(3, 3, kernel_size=1, stride=1, bias=False)
    # a.zero_grad()
    # print(a.weight.grad)
    import torch
    PRETRAINED_PATH='/home/admin/code/cassle_official/experiments/2022_07_13_17_37_12-simclr-cifar100/2cpkm0r5/simclr-cifar100-task0-ep=499-2cpkm0r5.ckpt'
    state_dict = torch.load(PRETRAINED_PATH, map_location="cpu")["state_dict"]
    get_modified_state_dict(state_dict=state_dict)
