from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter
import pandas as pd



LABELNAMES= [ 'ADI', 'BACK', 'LYM', 'STR', 'DEB', 'MUC', 'TUM','MUS','NORM']
def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

def cutout(img, num_holes=8, length=28):
        """
        Args:
        img (Tensor): Tensor image of size (H, W, C). input is an image
        Returns:
        Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.shape[1]
        w = img.shape[2]
        c = img.shape[0]
        mask = np.ones([h, w], np.float32)
        for _ in range(num_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(max(0, y - length // 2), 0, h)
            y2 = np.clip(max(0, y + length // 2), 0, h)
            x1 = np.clip(max(0, x - length // 2), 0, w)
            x2 = np.clip(max(0, x + length // 2), 0, w)
            mask[y1: y2, x1: x2] = 0
        mask = np.expand_dims(mask, 0)
        mask = torch.from_numpy(mask)
        
        mask = torch.cat((mask,mask,mask), dim=0)
        img = img * mask

        return img


class SubsetTrainDataset(Dataset):
    def __init__(self, root_dir,base_dataset,dataset_csv_path, indices, mode, prob=None, transform=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.mode = mode
        self.prob = prob
        self.transform = transform
        self.root_dir=root_dir
        self.dataset_csv_path=dataset_csv_path
        gold_csv_path= os.path.join(self.dataset_csv_path, 'Train.csv')
        gold_file = pd.read_csv(gold_csv_path)  
        self.golds=gold_file.iloc[:, 1:].values.astype(int)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        global_idx = self.indices[i] 
        _, _,img_item,  label,_ ,_,_= self.base_dataset[global_idx]
        image_name = os.path.join(self.root_dir, 'train',img_item+'.tif')
        raw_img = Image.open(image_name).convert('RGB')
        # 两次增强（DivideMix 用）
        img1 = self.transform(raw_img)
        img2 = self.transform(raw_img)
        img1= cutout(img1)
        img2= cutout(img2)
        gold=self.golds[global_idx]  
        if self.mode == 'labeled':
            w_x = self.prob[global_idx]
            return img1, img2, label.argmax(), w_x,gold
        elif self.mode == 'unlabeled':
            return img1, img2,gold
        
    

     
class SubsetTrainDataloader():  
    def __init__(self,batch_size, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform_train = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                     transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0)),
                    transforms.RandomGrayscale(p=0.3),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])
                ])
    




    # dataloader/dataloader.py 里 SubsetTrainDataloader.run()

    def run(self, root_dir, GMM_dataset, dataset_csv_path,
            global_labeled_indices, global_unlabeled_indices, prob=[],
            labeled_batch_sampler=None):

        labeled_dataset = SubsetTrainDataset(
            root_dir=root_dir, base_dataset=GMM_dataset, dataset_csv_path=dataset_csv_path,
            indices=global_labeled_indices, mode='labeled', prob=prob,
            transform=self.transform_train
        )

        if labeled_batch_sampler is None:
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers
            )
        else:
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_sampler=labeled_batch_sampler,
                num_workers=self.num_workers
            )

        unlabeled_dataset = SubsetTrainDataset(
            root_dir=root_dir, base_dataset=GMM_dataset, dataset_csv_path=dataset_csv_path,
            indices=global_unlabeled_indices, mode='unlabeled', prob=[],
            transform=self.transform_train
        )
        unlabeled_trainloader = DataLoader(
            dataset=unlabeled_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

        return labeled_trainloader, unlabeled_trainloader


class BaseDataset(Dataset): 
    def __init__(self, dataset, r, root_dir, transform, num_class,mode,dataset_csv_path,noise_file=''): 
        super(Dataset, self).__init__()
        self.r = r # noise ratio
        self.transform = transform
        self.mode = mode  
        self.root_dir=root_dir
        self.num_class=num_class
        self.dataset_csv_path=dataset_csv_path
        self.noise_file=noise_file
        if self.mode=='test':
            test_csv_path= os.path.join(self.dataset_csv_path, 'Test.csv')
            file = pd.read_csv(test_csv_path)                 # CSV: image,MEL,NV,BCC,AKIEC,BKL,DF,VASC
        else:# 训练集要读取有标签噪声的
            if not os.path.exists(self.noise_file): 
                self.getNoisyLabels()
            print(f'| Noisy file already exist {self.noise_file}')   
            file = pd.read_csv(self.noise_file)  
            gold_csv_path= os.path.join(self.dataset_csv_path, 'Train.csv')
            gold_file = pd.read_csv(gold_csv_path)  
            self.golds=gold_file.iloc[:, 1:].values.astype(int)
        self.root_dir = root_dir
        self.images = file['image'].values
        self.labels = file.iloc[:, 1:].values.astype(int)
        self.transform = transform
        print('Total_{}# images:{}, labels:{}'.format(self.mode,len(self.images),len(self.labels)))
   
                
    def __getitem__(self, index): # 需要Images和labels
        if self.mode=='test':
            items = self.images[index]
            image_name = os.path.join(self.root_dir,'test', self.images[index]+'.tif')
            # image_name = os.path.join(self.root_dir, 'Training_Input',self.images[index]+'.jpg')
            img = Image.open(image_name).convert('RGB')
            img = self.transform(img) 
            target= self.labels[index]
            return items, index, img, target
        else:# GMM warmup 
            image_name = os.path.join(self.root_dir, 'train',self.images[index]+'.tif')
            raw_img = Image.open(image_name).convert('RGB')
            img1 = self.transform(raw_img) 
            img1=cutout(img1)
            # img2 = self.transform(raw_img) 
            # img2=cutout(img2)
            target= self.labels[index]  
            gold=self.golds[index]  
            return img1,img1, self.images[index],target, index,gold,image_name 
           
    def __len__(self):
        if self.mode!='test':
            return len(self.images)
        else:
            return len(self.images)         
     
    def getNoisyLabels(self):
        train_label=[]
        train_csv_path = os.path.join(self.dataset_csv_path, 'Train.csv')
        file = pd.read_csv(train_csv_path)  # CSV: image,MEL,NV,BCC,AKIEC,BKL,DF,VASC
        img_ids = file['image'].values
        label_mat = file[LABELNAMES ].values.astype(int)          # (N,7)
        train_labels = np.argmax(label_mat, axis=1).astype(int)   # (N,)
        train_label = train_labels.tolist()                        # list[int]
        self.num_train = len(train_label) 
        noise_label = []
        # ---------- 噪声转移概率分支（gammaNoise） ----------
        # P(tilde_y = y) = 1 - r
        # 若发生噪声，则 j!=i 按类别频数 N_j / (N - N_i) 采样，更偏向多数类
        counts = [0 for _ in range(self.num_class)]# 统计每个类别的样本数 N_j
        for y in train_label:
            counts[int(y)] += 1
        for i in range(self.num_train):
            yi = int(train_label[i])
            # 以 1-r的概率保持原标签
            if random.random() > self.r:# 无噪声
                noise_label.append(yi)
            else:
                # 发生噪声：从 j != yi 中按 N_j / (N - N_yi) 采样
                cand = []# 候选的“错误类别集合”
                w = [] #每个候选类别对应的权重（出现频次）
                for j in range(self.num_class):
                    if j == yi:
                        continue
                    cand.append(j)
                    w.append(counts[j])
                # 归一化权重为概率
                s = float(sum(w))
                if s == 0:
                    # 极端兜底：若权重全为 0，退化为均匀采样（基本不会发生）
                    noiselabel = random.choice(cand)
                else:
                    p = [wj / s for wj in w]
                    noiselabel = int(np.random.choice(cand, p=p))
                noise_label.append(noiselabel)
        # noise_label: List[int] -> one-hot
        noise_label = np.array(noise_label, dtype=int)          # (N,)
        one_hot = np.zeros((len(noise_label), self.num_class), dtype=int)
        one_hot[np.arange(len(noise_label)), noise_label] = 1                  # (N,7)
        # 构造 DataFrame（与 ISIC 官方 CSV 完全同构）
        noise_df = pd.DataFrame(one_hot, columns=LABELNAMES )
        noise_df.insert(0, 'image', img_ids)                     # 第一列 image
        # 确保目录存在
        os.makedirs(os.path.dirname(self.noise_file), exist_ok=True)
        # 写入 CSV
        noise_df.to_csv(self.noise_file, index=False)
        print(f'| Noisy labels saved to {self.noise_file}')   
        
     
class  BaseDataloader():  
    def __init__(self, dataset, r, batch_size, num_workers, root_dir,num_class,dataset_csv_path,seed,noise_file='' ):
        self.dataset = dataset
        self.r = r
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.noise_file = noise_file
        self.seed=seed
        self.num_class=num_class
        self.dataset_csv_path=dataset_csv_path
        self.transform_train = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomAffine(degrees=10, translate=(0.02, 0.02)),
                    transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0)),
                transforms.RandomGrayscale(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
            ])
        self.transform_test = transforms.Compose([
                                            transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                    [0.229, 0.224, 0.225])])     
    def run(self,mode):
        if mode=='warmup':
            all_dataset = BaseDataset(dataset=self.dataset,r=self.r,\
                                        root_dir=self.root_dir, transform=self.transform_train,\
                                        num_class=self.num_class, mode="warmup",dataset_csv_path=self.dataset_csv_path,noise_file=self.noise_file)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
        elif mode=='test':
            test_dataset = BaseDataset(dataset=self.dataset,r=self.r, \
                                         root_dir=self.root_dir, transform=self.transform_test,\
                                         num_class=self.num_class,mode='test',dataset_csv_path=self.dataset_csv_path)      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='GMM_train':
            GMM_dataset = BaseDataset(dataset=self.dataset, r=self.r, \
                                         root_dir=self.root_dir, transform=self.transform_test,
                                         num_class=self.num_class, mode='GMM_train',dataset_csv_path=self.dataset_csv_path,\
                                         noise_file=self.noise_file)      
            GMM_loader = DataLoader(
                dataset=GMM_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return GMM_loader,GMM_dataset            
        
        
        