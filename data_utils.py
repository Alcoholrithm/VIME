import pandas as pd
from typing import Dict, Any, List
from numpy.typing import NDArray

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler, SequentialSampler, RandomSampler
from pytorch_lightning import LightningDataModule

from catalyst.data.sampler import BalanceClassSampler, DistributedSamplerWrapper

from misc.utils import *

class VIMESelfDataset(Dataset):
    def __init__(self, X: pd.DataFrame, data_hparams: Dict[str, Any], continous_cols: List = None, category_cols: List = None):
        
        

        self.cont_data = torch.FloatTensor(X[continous_cols].values)
        self.cat_data = torch.FloatTensor(X[category_cols].values)
        
        self.continuous_cols = continous_cols
        self.category_cols = category_cols
        
        self.data_hparams = data_hparams



    def __getitem__(self, idx):
        # the dataset must return a pair of samples: the anchor and a random one from the
        # dataset that will be used to corrupt the anchor
        cat_samples = self.cat_data[idx]
        m_unlab = mask_generator(self.data_hparams["p_m"], cat_samples)
        cat_m_label, cat_x_tilde = pretext_generator(m_unlab, cat_samples, self.cat_data)
        
        cont_samples = self.cont_data[idx]
        m_unlab = mask_generator(self.data_hparams["p_m"], cont_samples)
        cont_m_label, cont_x_tilde = pretext_generator(m_unlab, cont_samples, self.cont_data)

        m_label = torch.concat((cat_m_label, cont_m_label)).float()
        x_tilde = torch.concat((cat_x_tilde, cont_x_tilde)).float()

        x = torch.concat((cat_samples, cont_samples))
        
        return {
                "input" : x_tilde,
                "label" : (m_label, x)
                }

    def __len__(self):
        return len(self.cat_data)
    
class VIMEClassificationDataset(Dataset):
    def __init__(self, X: pd.DataFrame, Y: NDArray[np.int_], data_hparams: Dict[str, Any], unlabeled_data: pd.DataFrame = None, continous_cols: List = None, category_cols: List = None, u_label = -1, is_test: bool = False):
        
        if unlabeled_data is not None:
            X = X.append(unlabeled_data)
            
        self.cont_data = torch.FloatTensor(X[continous_cols].values)
        self.cat_data = torch.FloatTensor(X[category_cols].values)
        
        self.continuous_cols = continous_cols
        self.category_cols = category_cols
        
        self.u_label = u_label
        self.is_test = is_test
        
        if is_test is False:
            self.data_hparams = data_hparams
        
            self.label = torch.LongTensor(Y)
        
            if unlabeled_data is not None:
                self.label = torch.concat((self.label, torch.LongTensor([-1 for _ in range(len(unlabeled_data))])), dim=0)
                
            class_counts = [sum((self.label == i)) for i in set(self.label.numpy())]
            num_samples = len(self.label)

            class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
            self.weights = [class_weights[self.label[i]] for i in range(int(num_samples))]
            
    def generate_x_tildes(self, cat_samples, cont_samples):
        m_unlab = mask_generator(self.data_hparams["p_m"], cat_samples)
        dcat_m_label, cat_x_tilde = pretext_generator(m_unlab, cat_samples, self.cat_data)
        
        m_unlab = mask_generator(self.data_hparams["p_m"], cont_samples)
        cont_m_label, cont_x_tilde = pretext_generator(m_unlab, cont_samples, self.cont_data)
        x_tilde = torch.concat((cat_x_tilde, cont_x_tilde)).float()
        
        return x_tilde

    def __getitem__(self, idx):
        # the dataset must return a pair of samples: the anchor and a random one from the
        # dataset that will be used to corrupt the anchor
        cat_samples = self.cat_data[idx]
        cont_samples = self.cont_data[idx]
        x = torch.concat((cat_samples, cont_samples)).squeeze()
        if self.is_test is False:
            
            if self.label[idx] == self.u_label:
                xs = [x]
                
                xs.extend([self.generate_x_tildes(cat_samples, cont_samples) for _ in range(self.data_hparams["K"])])

                xs = torch.stack(xs)
                return {
                    "input" : xs,
                    "label" : torch.LongTensor([self.u_label for _ in range(len(xs))])
                }
            else:
                return {
                    "input" : x.unsqueeze(0),
                    "label" : self.label[idx].unsqueeze(0)
                }
        else:
            return {
                    "input" : x,
                    "label" : self.u_label,
            }

    def __len__(self):
        return len(self.cat_data)


class VIMERegressionDataset(Dataset):
    def __init__(self, X: pd.DataFrame, Y: NDArray[np.float_], data_hparams: Dict[str, Any], unlabeled_data: pd.DataFrame = None, continous_cols: List = None, category_cols: List = None, u_label = -1, is_test: bool = False):
        
        if unlabeled_data is not None:
            X = X.append(unlabeled_data)
            
        self.cont_data = torch.FloatTensor(X[continous_cols].values)
        self.cat_data = torch.FloatTensor(X[category_cols].values)
        
        self.continuous_cols = continous_cols
        self.category_cols = category_cols
        
        self.u_label = u_label
        self.is_test = is_test
        
        if is_test is False:
            self.data_hparams = data_hparams
        
            self.label = torch.FloatTensor(Y)
        
            if unlabeled_data is not None:
                self.label = torch.concat((self.label, torch.FloatTensor([u_label for _ in range(len(unlabeled_data))])), dim=0)
            
    def generate_x_tildes(self, cat_samples, cont_samples):
        m_unlab = mask_generator(self.data_hparams["p_m"], cat_samples)
        dcat_m_label, cat_x_tilde = pretext_generator(m_unlab, cat_samples, self.cat_data)
        
        m_unlab = mask_generator(self.data_hparams["p_m"], cont_samples)
        cont_m_label, cont_x_tilde = pretext_generator(m_unlab, cont_samples, self.cont_data)
        x_tilde = torch.concat((cat_x_tilde, cont_x_tilde)).float()
        
        return x_tilde

    def __getitem__(self, idx):
        # the dataset must return a pair of samples: the anchor and a random one from the
        # dataset that will be used to corrupt the anchor
        cat_samples = self.cat_data[idx]
        cont_samples = self.cont_data[idx]
        x = torch.concat((cat_samples, cont_samples)).squeeze()
        if self.is_test is False:
            
            if self.label[idx] == self.u_label:
                xs = [x]
                
                xs.extend([self.generate_x_tildes(cat_samples, cont_samples) for _ in range(self.data_hparams["K"])])

                xs = torch.stack(xs)
                return {
                    "input" : xs,
                    "label" : torch.FloatTensor([self.u_label for _ in range(len(xs))])
                }
            else:
                return {
                    "input" : x.unsqueeze(0),
                    "label" : self.label[idx].unsqueeze(0)
                }
        else:
            return {
                    "input" : x,
                    "label" : self.u_label,
            }

    def __len__(self):
        return len(self.cat_data)

class PLDataModule(LightningDataModule):
    def __init__(self, train_ds, val_ds, batch_size, n_gpus = 1, n_jobs = 32, drop_last = False, is_regression:bool = False):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds

        self.batch_size = batch_size

        self.n_gpus = n_gpus
        self.n_jobs = n_jobs
        self.is_regression = is_regression
        self.drop_last = drop_last

    def setup(self, stage: str):

        # if hasattr(self.train_ds, "label") and self.train_ds.label is None:
        #     sampler = SequentialSampler(self.train_ds)
        # elif self.is_regression or not hasattr(self.train_ds, "label"):
        #     sampler = RandomSampler(self.train_ds, num_samples = len(self.train_ds))
        # else:
        #     sampler = BalanceClassSampler(self.train_ds.label, mode="upsampling") 
            
        # if self.n_gpus > 1:
        #     sampler = DistributedSamplerWrapper(sampler)
        
        if hasattr(self.train_ds, 'label') and self.train_ds.label is None:
            sampler = SequentialSampler(self.train_ds)
        elif not hasattr(self.train_ds, 'label'):
            sampler = RandomSampler(self.train_ds, num_samples = len(self.train_ds))
        else:
            sampler = WeightedRandomSampler(self.train_ds.weights, num_samples = len(self.train_ds))
                # sampler = WeightedRandomSampler(self.train_ds.weights, num_samples = len(self.train_ds))

        def collate_fn(batch):
            # for x in batch:
            #     print(x["input"], " ||||||||||||||||||||||||||||| ", x["label"])
            #     print("====================================================\n\n\n\n")
            return {
                'input': torch.concat([x['input'] for x in batch], dim=0),
                'label': torch.concat([x['label'] for x in batch], dim=0)
            }

        if not hasattr(self.train_ds, "label"):#  or not self.is_regression:
            collate_fn = None

        self.train_dl = DataLoader(self.train_ds, 
                                   batch_size = self.batch_size, 
                                   shuffle=False, 
                                   sampler = sampler,
                                   num_workers=self.n_jobs,
                                   drop_last=self.drop_last,
                                   collate_fn = collate_fn)
        self.val_dl = DataLoader(self.val_ds, batch_size = self.batch_size, shuffle=False, sampler = SequentialSampler(self.val_ds), num_workers=self.n_jobs, drop_last=False)
    
    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl
# class VIMEDataset(Dataset):
#     def __init__(self, X: pd.DataFrame, Y: NDArray[np.int_], data_hparams: Dict[str, Any], unlabeled_data: pd.DataFrame = None, continous_cols: List = None, category_cols: List = None, is_test: bool = False):
        
#         if unlabeled_data is not None:
#             X = X.append(unlabeled_data)
            
#         self.cont_data = torch.FloatTensor(X[continous_cols].values)
#         self.cat_data = torch.FloatTensor(X[category_cols].values)
        
#         self.continuous_cols = continous_cols
#         self.category_cols = category_cols
        
#         self.is_test = is_test
        
#         if is_test is False:
#             self.data_hparams = data_hparams
        
#             self.label = torch.LongTensor(Y)
        
#             class_counts = [sum((Y == i)) for i in set(Y)]
#             num_samples = len(Y)

#             class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
#             self.weights = [class_weights[Y[i]] for i in range(int(num_samples))]
        
#             if unlabeled_data is not None:
#                 self.label = torch.concat((self.label, torch.LongTensor([-1 for _ in range(len(unlabeled_data))])), dim=0)
            
#     def generate_x_tildes(self, cat_samples, cont_samples):
#         m_unlab = mask_generator(self.data_hparams["p_m"], cat_samples)
#         dcat_m_label, cat_x_tilde = pretext_generator(m_unlab, cat_samples, self.cat_data)
        
#         m_unlab = mask_generator(self.data_hparams["p_m"], cont_samples)
#         cont_m_label, cont_x_tilde = pretext_generator(m_unlab, cont_samples, self.cont_data)
#         x_tilde = torch.concat((cat_x_tilde, cont_x_tilde)).float()
        
#         return x_tilde

#     def __getitem__(self, idx):
#         # the dataset must return a pair of samples: the anchor and a random one from the
#         # dataset that will be used to corrupt the anchor
#         cat_samples = self.cat_data[idx]
#         cont_samples = self.cont_data[idx]
#         x = torch.concat((cat_samples, cont_samples)).squeeze()
#         if self.is_test is False:
            
#             if self.label[idx] == -1:
#                 xs = [x]
                
#                 xs.extend([self.generate_x_tildes(cat_samples, cont_samples) for _ in range(self.data_hparams["K"])])

#                 xs = torch.concat(xs)
#                 return {
#                     "input" : xs,
#                     "label" : None
#                 }
#             else:
#                 return {
#                     "input" : x,
#                     "label" : self.label[idx]
#                 }
#         else:
#             return {
#                     "input" : x,
#                     "label" : -1,
#             }

#     def __len__(self):
#         return len(self.cat_data)
    

# class PLDataModule(LightningDataModule):
#     def __init__(self, train_ds, val_ds, batch_size, n_gpus = 1, n_jobs = 32, drop_last = False, is_regression:bool = False):
#         super().__init__()
#         self.train_ds = train_ds
#         self.val_ds = val_ds

#         self.batch_size = batch_size

#         self.n_gpus = n_gpus
#         self.n_jobs = n_jobs
#         self.is_regression = is_regression
#         self.drop_last = drop_last

#     def setup(self, stage: str):

#         # if hasattr(self.train_ds, "label") and self.train_ds.label is None:
#         #     sampler = SequentialSampler(self.train_ds)
#         # elif self.is_regression or not hasattr(self.train_ds, "label"):
#         #     sampler = RandomSampler(self.train_ds, num_samples = len(self.train_ds))
#         # else:
#         #     sampler = BalanceClassSampler(self.train_ds.label, mode="upsampling") 
            
#         # if self.n_gpus > 1:
#         #     sampler = DistributedSamplerWrapper(sampler)
            
#         if self.n_gpus > 1:
#             sampler = BalanceClassSampler(self.train_ds.labels, mode="upsampling")
#             sampler = DistributedSamplerWrapper(sampler)
#         else:
#             if not hasattr(self.train_ds, 'label'):
#                 sampler = RandomSampler(self.train_ds, num_samples = len(self.train_ds))
#             elif self.train_ds.label is None:
#                 sampler = SequentialSampler(self.train_ds)
#             else:
#                 sampler = WeightedRandomSampler(self.train_ds.weights, num_samples = len(self.train_ds))

#         # def collate_fn(batch):
#         #     for x in batch:
#         #         print(x["input"].shape, x["label"].shape)
#         #     return {
#         #         'input': torch.concat([x['input'] for x in batch], dim=0),
#         #         'label': torch.concat([x['label'] for x in batch], dim=0).squeeze()
#         #     }

#         # if not hasattr(self.train_ds, "label"):
#         collate_fn = None

#         self.train_dl = DataLoader(self.train_ds, 
#                                    batch_size = self.batch_size, 
#                                    shuffle=False, 
#                                    sampler = sampler,
#                                    num_workers=self.n_jobs,
#                                    drop_last=self.drop_last,
#                                    collate_fn = collate_fn)
#         self.val_dl = DataLoader(self.val_ds, batch_size = self.batch_size, shuffle=False, sampler = SequentialSampler(self.val_ds), num_workers=self.n_jobs, drop_last=False)
    
#     def train_dataloader(self):
#         return self.train_dl

#     def val_dataloader(self):
#         return self.val_dl