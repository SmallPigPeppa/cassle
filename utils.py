from models.resnet_modified import resnet18
def get_modified_state_dict(state_dict):
    import collections
    modified_state_dict = collections.OrderedDict()
    # modified_resnet=resnet18()
    # modified_resnet_keys=[i for i,_ in modified_resnet.named_parameters()]
    for key, value in state_dict.items():
        if key not in modified_resnet_keys and 'conv' in key and 'encoder' in key and 'layer' in key:

            a = key.split('.')
            key = f'{a[0]}.{a[1]}.{a[2]}.{a[3]}.conv2d_3x3.{a[4]}'
        modified_state_dict[key] = value
    return modified_state_dict

    state = torch.load(ckpt_path)["state_dict"]
    for k in list(state.keys()):
        if "encoder" in k:
            state[k.replace("encoder", "backbone")] = state[k]
            warnings.warn(
                "You are using an older checkpoint. Use a new one as some issues might arrise."
            )
        if "backbone" in k:
            state[k.replace("backbone.", "")] = state[k]
        del state[k]


if __name__=='__main__':
    # import torch.nn as nn
    # a=nn.Conv2d(3, 3, kernel_size=1, stride=1, bias=False)
    # a.zero_grad()
    # print(a.weight.grad)
    import torch
    no_expansion='/home/admin/code/cassle_official/experiments/2022_07_13_17_37_12-simclr-cifar100/2cpkm0r5/simclr-cifar100-task0-ep=499-2cpkm0r5.ckpt'
    use_expansion='/home/admin/code/cassle_v115.0/experiments/2022_08_11_21_41_19-simclr-cifar100/1la2bpsl/simclr-cifar100-task4-ep=499-1la2bpsl.ckpt'
    state1 = torch.load(no_expansion)["state_dict"]
    state2 = torch.load(use_expansion)["state_dict"]
    key1=[key for key,_ in state1.items()]
    key2=[key for key,_ in state2.items()]
    key3=[]
    key4=[]
    for k in key2:
        if k not in key1:
            print(k)
            key3.append(k)
    print(len(key3))

    for k in key1:
        if k not in key2:
            print(k)
            key4.append(k)
    print(len(key4))