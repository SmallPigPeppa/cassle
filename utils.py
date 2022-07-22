from models.resnet_modified import resnet50
def get_modified_state_dict(state_dict):
    import collections
    modified_state_dict = collections.OrderedDict()
    modified_resnet=resnet50()
    modified_resnet_keys=[i for i,_ in modified_resnet.named_parameters()]
    for key, value in state_dict.items():
        if key not in modified_resnet_keys and 'conv2' in key:
            a = key.split('.')
            key = f'{a[0]}.{a[1]}.{a[2]}.conv2d_3x3.{a[3]}'
        modified_state_dict[key] = value
    return modified_state_dict