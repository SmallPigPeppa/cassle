import torch
from tqdm import tqdm
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from cpn import PrototypeClassifier


def kmeans_filter(task_loader, pretrained_model, num_classes, dim_features=512, min_logits=0.5):
    # get all feats, idxs
    feats_task = []
    idxs_task = []
    for batch in tqdm(task_loader):
        idxs_i = batch[0]
        imgs_i = batch[1]
        for x in imgs_i:
            out = pretrained_model(x)
            z = out["feats"]
            feats_task.append(z.cpu().detach().numpy())
            idxs_i = torch.reshape(idxs_i, (-1,))
            idxs_task.append(idxs_i.cpu().detach().numpy())

    feats_task = np.vstack(feats_task)
    idxs_task = np.hstack(idxs_task)

    # kmeans on feats
    kmeans = KMeans(n_clusters=num_classes, random_state=0).fit(preprocessing.normalize(feats_task))
    print("kmeans.cluster_centers_.shape:", kmeans.cluster_centers_.shape)

    # filter feats by cosine cpn
    cosine_cpn = PrototypeClassifier(dim_features=dim_features, num_classes=num_classes,
                                     centers=preprocessing.normalize(kmeans.cluster_centers_))
    logits_task = cosine_cpn.logits(torch.tensor(preprocessing.normalize(feats_task)))

    max_logits, _ = torch.max(logits_task, 1)
    max_logits = max_logits.cpu().detach().numpy()

    invalid_idx = np.where(max_logits < min_logits)[0]
    kmeans_labels = kmeans.labels_
    kmeans_labels[invalid_idx] = -1
    # filter kmeans label differenet on image_view1 and image_view2
    kmeans_dict = dict(zip(idxs_task, kmeans_labels))
    kmeans_dict2 = dict(zip(idxs_task[::-1], kmeans_labels[::-1]))
    diff_keys = [k for k, _ in set(kmeans_dict.items()) - set(kmeans_dict2.items())]
    for k in diff_keys:
        kmeans_dict[k] = -1

    print("all samples:", len(kmeans_dict))
    print("valid samples:", len(kmeans_dict) - len(invalid_idx))
    print("valid rate:", (len(kmeans_dict) - len(invalid_idx)) / len(kmeans_dict))
    centers = preprocessing.normalize(kmeans.cluster_centers_)

    return kmeans_dict, preprocessing.normalize(kmeans.cluster_centers_)


# [mydict[x] for x in mykeys]
#  [c[x] for x in a.cpu().detach().numpy()]

def feats_centers(task_loader, pretrained_model, tasks):
    # get all feats, idxs
    feats_task = []
    labels_task = []
    centers_task = []
    for batch in tqdm(task_loader):
        idxs_i = batch[0]
        imgs_i = batch[1]
        labels_i = batch[2]
        for x in imgs_i:
            out = pretrained_model(x)
            z = out["feats"]
            feats_task.append(z.cpu().detach().numpy())
            idxs_i = torch.reshape(idxs_i, (-1,))
            labels_task.append(labels_i.cpu().detach().numpy())

    feats_task = np.vstack(feats_task)
    labels_task = np.hstack(labels_task)

    for i in tqdm(tasks):
        index_ci = np.where(labels_task == i)[0]
        feats_ci = feats_task[index_ci]
        centers_ci=np.average(feats_ci,axis=0)
        centers_task.append(centers_ci)
    centers_task = np.vstack(centers_task)
    return torch.tensor(centers_task)
