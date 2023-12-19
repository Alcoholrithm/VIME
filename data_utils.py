import pandas as pd
from typing import Dict, Any, List
from numpy.typing import NDArray

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler, SequentialSampler, RandomSampler
from pytorch_lightning import LightningDataModule

from misc.utils import *


class VIMESelfDataset(Dataset):
    """The dataset for the self-supervised learning of VIME
    """
    def __init__(self, X: pd.DataFrame, data_hparams: Dict[str, Any], continous_cols: List = None, category_cols: List = None):
        """Initialize the self-supervised learning dataset

        Args:
            X (pd.DataFrame): The features of the data
            data_hparams (Dict[str, Any]): The hyperparameters for mask_generator and pretext_generator
            continous_cols (List, optional): The list of continuous columns. Defaults to None.
            category_cols (List, optional): The list of categorical columns. Defaults to None.
        """
        self.cont_data = torch.FloatTensor(X[continous_cols].values)
        self.cat_data = torch.FloatTensor(X[category_cols].values)
        
        self.continuous_cols = continous_cols
        self.category_cols = category_cols
        
        self.data_hparams = data_hparams



    def __getitem__(self, idx: int):
        """Return a input and label pair

        Args:
            idx (int): The index of the data to sample

        Returns:
            Dict[str, Any]: A pair of input and label for self-supervised learning
        """
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
        """Return the length of the dataset
        """
        return len(self.cat_data)
    
class VIMEClassificationDataset(Dataset):
    """The classification dataset for the semi-supervised learning of VIME
    """
    def __init__(self, X: pd.DataFrame, Y: NDArray[np.int_], data_hparams: Dict[str, Any], unlabeled_data: pd.DataFrame = None, continous_cols: List = None, category_cols: List = None, u_label = -1, is_test: bool = False):
        """Initialize the semi-supervised learning dataset for the classification

        Args:
            X (pd.DataFrame): The features of the labeled data
            Y (NDArray[np.int_]): The label of the labeled data
            data_hparams (Dict[str, Any]): The hyperparameters for consistency regularization
            unlabeled_data (pd.DataFrame, optional): The features of the unlabeled data. Defaults to None.
            continous_cols (List, optional): The list of continuous columns. Defaults to None.
            category_cols (List, optional): The list of categorical columns. Defaults to None.
            u_label (int, optional): The specifier for unlabeled sample. Defaults to -1.
            is_test (bool, optional): The flag that determines whether the dataset is for testing or not. Defaults to False.
        """
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
                self.label = torch.concat((self.label, torch.LongTensor([self.u_label for _ in range(len(unlabeled_data))])), dim=0)
                
            class_counts = [sum((self.label == i)) for i in set(self.label.numpy())]
            num_samples = len(self.label)

            class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
            self.weights = [class_weights[self.label[i]] for i in range(int(num_samples))]
            
    def generate_x_tildes(self, cat_samples: torch.FloatTensor, cont_samples:torch.FloatTensor) -> torch.FloatTensor:
        """Generate x_tilde for consistency regularization

        Args:
            cat_samples (torch.FloatTensor): The categorical features to generate x_tilde
            cont_samples (torch.FloatTensor): The continuous features to generate x_tilde

        Returns:
            torch.FloatTensor: x_tilde for consistency regularization
        """
        m_unlab = mask_generator(self.data_hparams["p_m"], cat_samples)
        dcat_m_label, cat_x_tilde = pretext_generator(m_unlab, cat_samples, self.cat_data)
        
        m_unlab = mask_generator(self.data_hparams["p_m"], cont_samples)
        cont_m_label, cont_x_tilde = pretext_generator(m_unlab, cont_samples, self.cont_data)
        x_tilde = torch.concat((cat_x_tilde, cont_x_tilde)).float()
        
        return x_tilde

    def __getitem__(self, idx):
        """Return a input and label pair

        Args:
            idx (int): The index of the data to sample

        Returns:
            Dict[str, Any]: A pair of input and label for semi-supervised learning
        """
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
        """Return the length of the dataset
        """
        return len(self.cat_data)


class VIMERegressionDataset(Dataset):
    """The regression dataset for the semi-supervised learning of VIME
    """
    def __init__(self, X: pd.DataFrame, Y: NDArray[np.float_], data_hparams: Dict[str, Any], unlabeled_data: pd.DataFrame = None, continous_cols: List = None, category_cols: List = None, u_label = -1, is_test: bool = False):
        """Initialize the semi-supervised learning dataset for the regression

        Args:
            X (pd.DataFrame): The features of the labeled data
            Y (NDArray[np.int_]): The label of the labeled data
            data_hparams (Dict[str, Any]): The hyperparameters for consistency regularization
            unlabeled_data (pd.DataFrame, optional): The features of the unlabeled data. Defaults to None.
            continous_cols (List, optional): The list of continuous columns. Defaults to None.
            category_cols (List, optional): The list of categorical columns. Defaults to None.
            u_label (int, optional): The specifier for unlabeled sample. Defaults to -1.
            is_test (bool, optional): The flag that determines whether the dataset is for testing or not. Defaults to False.
        """
        
        self.weights = [1.0 for _ in range(len(X))]
        
        if unlabeled_data is not None:
            unlabeled_weight = len(X) / len(unlabeled_data)
            self.weights.extend([unlabeled_weight for _ in range(len(unlabeled_data))])
            
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
                
            
    def generate_x_tildes(self, cat_samples: torch.FloatTensor, cont_samples:torch.FloatTensor) -> torch.FloatTensor:
        """Generate x_tilde for consistency regularization

        Args:
            cat_samples (torch.FloatTensor): The categorical features to generate x_tilde
            cont_samples (torch.FloatTensor): The continuous features to generate x_tilde

        Returns:
            torch.FloatTensor: x_tilde for consistency regularization
        """
        m_unlab = mask_generator(self.data_hparams["p_m"], cat_samples)
        dcat_m_label, cat_x_tilde = pretext_generator(m_unlab, cat_samples, self.cat_data)
        
        m_unlab = mask_generator(self.data_hparams["p_m"], cont_samples)
        cont_m_label, cont_x_tilde = pretext_generator(m_unlab, cont_samples, self.cont_data)
        x_tilde = torch.concat((cat_x_tilde, cont_x_tilde)).float()
        
        return x_tilde

    def __getitem__(self, idx):
        """Return a input and label pair

        Args:
            idx (int): The index of the data to sample

        Returns:
            Dict[str, Any]: A pair of input and label for semi-supervised learning
        """
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
    """The pytorch lightning datamodule for VIME
    """
    def __init__(self, train_ds:Dataset, val_ds:Dataset, batch_size: int, n_gpus: int = 1, n_jobs: int = 32, drop_last: int = False, is_regression:bool = False):
        """Initialize the datamodule

        Args:
            train_ds (Dataset): The training dataset
            val_ds (Dataset): The validation dataset
            batch_size (int): The batch size of the dataset
            n_gpus (int, optional): The number of the gpus to use. Defaults to 1.
            n_jobs (int, optional): The number of the cpu core to use. Defaults to 32.
            drop_last (bool, optional): The flag to drop the last batch or not. Defaults to False.
            is_regression (bool, optional): The flag that determines whether the datamodule is for regression task or not. Defaults to False.
        """
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds

        self.batch_size = batch_size

        self.n_gpus = n_gpus
        self.n_jobs = n_jobs
        self.is_regression = is_regression
        self.drop_last = drop_last

    def setup(self, stage: str):
        """Setup the datamodule for pytorch lightning module of VIME
        
        Use a weighted random sampler for the finetunning step of the classification task, otherwise use a random sampler.
        
        Args:
            stage (str): For compatibility, do not use
        """
        if hasattr(self.train_ds, 'label') and self.train_ds.label is None:
            sampler = SequentialSampler(self.train_ds)
        elif not hasattr(self.train_ds, 'label'):
            sampler = RandomSampler(self.train_ds, num_samples = len(self.train_ds))
        else:
            sampler = WeightedRandomSampler(self.train_ds.weights, num_samples = len(self.train_ds))

        def collate_fn(batch):
            return {
                'input': torch.concat([x['input'] for x in batch], dim=0),
                'label': torch.concat([x['label'] for x in batch], dim=0)
            }

        if not hasattr(self.train_ds, "label"):
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
        """Return the training dataloader
        """
        return self.train_dl

    def val_dataloader(self):
        """Return the validation dataloader
        """
        return self.val_dl