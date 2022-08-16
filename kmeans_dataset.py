from torch.utils.data import Dataset


class KmeansDataset(Dataset):

    def __init__(self, task_loader, pretrained_model, tasks, dimfeatures=512):
        print(next(iter(train_loaders["task0"])))

        # 正常输出为 idx,[image1,image2],label
        # 希望输出转换为 idx,[image1,image2],label,KNN-label
        feats_task = []
        labels_task = []
        from tqdm import tqdm
        import numpy as np
        for batch in tqdm(task_loader):
            imgs1 = batch[1][0]
            imgs2 = batch[1][1]
            labels = batch[2]
            out1 = pretrained_model(imgs1)
            z_i = out1["feats"]
            feats_task.append(z_i.cpu().detach().numpy())
            labels_task.append(labels.cpu().detach().numpy())
            # break
        feats_task = np.vstack(feats_task)
        labels_task = np.hstack(labels_task)
        from sklearn import preprocessing
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=len(tasks[args.task_idx]), random_state=0).fit(preprocessing.normalize(feats_task))
        from cpn import PrototypeClassifier
        print("kmeans.cluster_centers_.shape:", kmeans.cluster_centers_.shape)
        m_cpn = PrototypeClassifier(dim_features=512, num_classes=len(tasks[args.task_idx]),
                                    centers=preprocessing.normalize(kmeans.cluster_centers_))
        logits_task = m_cpn.logits(torch.tensor(preprocessing.normalize(feats_task)))
        # logits_all_kmeans=logits_all_kmeans.cpu().detach().numpy()
        max_logits, _ = torch.max(logits_task, 1)
        max_logits = max_logits.cpu().detach().numpy()
        # valid_mask = np.where(max_logits >= 0.5)[0]
        no_used_smples = np.where(max_logits < 0.5)[0]
        kmeans_labels = kmeans.labels_
        kmeans_labels[no_used_smples] = -1

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
