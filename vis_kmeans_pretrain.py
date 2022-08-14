import os
from pprint import pprint
import types

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from cassle.args.setup import parse_args_pretrain
from cassle.methods import METHODS
from cassle.distillers import DISTILLERS

try:
    from cassle.methods.dali import PretrainABC
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True

try:
    from cassle.utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True

from cassle.utils.checkpointer import Checkpointer
from cassle.utils.classification_dataloader import prepare_data as prepare_data_classification
from cassle.utils.pretrain_dataloader import (
    prepare_dataloader,
    prepare_datasets,
    prepare_multicrop_transform,
    prepare_n_crop_transform,
    prepare_transform,
    split_dataset,
)


def main():
    seed_everything(5)

    args = parse_args_pretrain()

    # online eval dataset reloads when task dataset is over
    args.multiple_trainloader_mode = "min_size"

    # set online eval batch size and num workers
    args.online_eval_batch_size = int(args.batch_size) if args.dataset == "cifar100" else None

    # split classes into tasks
    tasks = None
    if args.split_strategy == "class":
        assert args.num_classes % args.num_tasks == 0
        tasks = torch.tensor(list(range(args.num_classes))).chunk(args.num_tasks)

    # pretrain and online eval dataloaders
    if not args.dali:
        # asymmetric augmentations
        if args.unique_augs > 1:
            transform = [
                prepare_transform(args.dataset, multicrop=args.multicrop, **kwargs)
                for kwargs in args.transform_kwargs
            ]
        else:
            transform = prepare_transform(
                args.dataset, multicrop=args.multicrop, **args.transform_kwargs
            )

        if args.debug_augmentations:
            print("Transforms:")
            pprint(transform)

        if args.multicrop:
            assert not args.unique_augs == 1

            if args.dataset in ["cifar10", "cifar100"]:
                size_crops = [32, 24]
            elif args.dataset == "stl10":
                size_crops = [96, 58]
            # imagenet or custom dataset
            else:
                size_crops = [224, 96]

            transform = prepare_multicrop_transform(
                transform, size_crops=size_crops, num_crops=[args.num_crops, args.num_small_crops]
            )
        else:
            if args.num_crops != 2:
                assert args.method == "wmse"

            online_eval_transform = transform[-1] if isinstance(transform, list) else transform
            task_transform = prepare_n_crop_transform(transform, num_crops=args.num_crops)

        train_dataset, online_eval_dataset = prepare_datasets(
            args.dataset,
            task_transform=task_transform,
            online_eval_transform=online_eval_transform,
            data_dir=args.data_dir,
            train_dir=args.train_dir,
            no_labels=args.no_labels,
        )

        task_dataset, tasks = split_dataset(
            train_dataset,
            tasks=tasks,
            task_idx=args.task_idx,
            num_tasks=args.num_tasks,
            split_strategy=args.split_strategy,
        )

        task_loader = prepare_dataloader(
            task_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        train_loaders = {f"task{args.task_idx}": task_loader}

        if args.online_eval_batch_size:
            online_eval_loader = prepare_dataloader(
                online_eval_dataset,
                batch_size=args.online_eval_batch_size,
                num_workers=args.num_workers,
            )
            train_loaders.update({"online_eval": online_eval_loader})

    # normal dataloader for when it is available
    if args.dataset == "custom" and (args.no_labels or args.val_dir is None):
        val_loader = None
    elif args.dataset in ["imagenet100", "imagenet"] and args.val_dir is None:
        val_loader = None
    else:
        _, val_loader = prepare_data_classification(
            args.dataset,
            data_dir=args.data_dir,
            train_dir=args.train_dir,
            val_dir=args.val_dir,
            batch_size=args.batch_size,
            num_workers=2 * args.num_workers,
        )

    # check method
    assert args.method in METHODS, f"Choose from {METHODS.keys()}"
    # build method
    MethodClass = METHODS[args.method]
    if args.dali:
        assert (
            _dali_avaliable
        ), "Dali is not currently avaiable, please install it first with [dali]."
        MethodClass = types.new_class(f"Dali{MethodClass.__name__}", (PretrainABC, MethodClass))

    if args.distiller:
        MethodClass = DISTILLERS[args.distiller](MethodClass)

    model = MethodClass(**args.__dict__, tasks=tasks if args.split_strategy == "class" else None)

    # only one resume mode can be true
    assert [args.resume_from_checkpoint, args.pretrained_model].count(True) <= 1

    if args.resume_from_checkpoint:
        pass  # handled by the trainer
    elif args.pretrained_model:
        print(f"Loading previous task checkpoint {args.pretrained_model}...")
        state_dict = torch.load(args.pretrained_model, map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict, strict=False)

    # visualize feats on task1
    # from tsne_torch import TorchTSNE as TSNE
    from sklearn.manifold import TSNE
    feats_all=[]
    labels_all=[]
    from tqdm import tqdm
    import numpy as np
    for batch in tqdm(task_loader):
    # for batch in tqdm(val_loader):
        imgs=batch[1][0]
        labels=batch[2]
        # imgs=batch[0]
        # labels=batch[1]
        # print(len(batch))
        # print(imgs.shape)
        # print(labels.shape)
        out_i=model(imgs)
        z_i=out_i["feats"]
        feats_all.append(z_i.cpu().detach().numpy())
        labels_all.append(labels.cpu().detach().numpy())
        # break
    feats_all = np.vstack(feats_all)
    labels_all = np.hstack(labels_all)

    print(feats_all.shape)
    print(labels_all.shape)

    feats_emb = TSNE(n_components=2, learning_rate='auto',init = 'random', perplexity = 3).fit_transform(feats_all)
    # # plot on each class
    import matplotlib.pyplot as plt
    #
    feats_all_kmeans=[]
    labels_all_kmeans=[]
    for i in range(20,25):
        index_ci = np.where(labels_all == i)[0]
        feats_ci = feats_all[index_ci]
        feats_all_kmeans.append(feats_ci)
        labels_all_kmeans.append([i]*len(feats_ci))
    feats_all_kmeans = np.vstack(feats_all_kmeans)
    labels_all_kmeans = np.hstack(labels_all_kmeans)

    from sklearn import preprocessing
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=5, random_state=0).fit(preprocessing.normalize(feats_all_kmeans))
    print(str([i for i in kmeans.labels_]))
    print(len(kmeans.labels_))
    dict={'3':20,'4':21,'0':22,'1':23,'2':24}
    right=0
    for i,x in enumerate(kmeans.labels_):
        if labels_all_kmeans[i]==dict[str(x)]:
            right+=1
    print('acc:',right/len(kmeans.labels_))


    # #
    # # print(feats_all2.shape,labels_all2.shape)
    # # # feats_emb = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(
    # # #     feats_all2)
    # # feats_emb = TSNE(n_components=2, learning_rate='auto',init = 'random', perplexity = 3).fit_transform(feats_all2)
    #
    # for i in tqdm(range(20,25)):
    #     index_ci = np.where(labels_all == i)[0]
    #     feats_emb_ci = feats_emb[index_ci]
    #     # feats_ci_emb = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(
    #     #     feats_ci)  # returns shape (n_samples, 2)
    #     plt.scatter(feats_emb_ci[:, 0], feats_emb_ci[:, 1], label=f'class-{i}')
    #
    # plt.legend()
    # # plt.show()
    # plt.savefig('/home/admin/result.pdf')
    #
    #
    # # X = ...  # shape (n_samples, d)
    # # X_emb = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(
    # #     X)  # returns shape (n_samples, 2)
if __name__ == "__main__":
    main()
