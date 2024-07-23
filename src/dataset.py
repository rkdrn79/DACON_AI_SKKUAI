from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from typing import Tuple, Any
import torch
import pickle

from src.preprocessing_factory.vaildation_index import get_validation_index
from src.preprocessing_factory.get_librosa import get_librosa_feature
from src.preprocessing_factory.augmentation import AudioAugmentation
from src.preprocessing_factory.feature_extractor import FeatureExtractor

def get_dataset(args) -> Tuple[Dataset,Dataset,Dataset, Any]:
    """
    train, valid, test dataset를 처리 후 반환합니다.
    """

    train_path = os.path.join(args.data_path, "./train.csv")
    train_df = pd.read_csv(train_path)
    
    validation_index = get_validation_index(args,train_df)

    valid_df = train_df.loc[validation_index]
    train_df = train_df.drop(validation_index)

    test_path = os.path.join(args.data_path, './test.csv')
    test_df = pd.read_csv(test_path)
    
    print(f"# of train: {len(train_df)}, # of valid: {len(valid_df)}")

    
    if args.is_inference: #used when inference
        train_dataset = None 
        valid_dataset = None 
        test_dataset = AudioDataset(args,test_df, test = True)
    else:
        train_dataset = AudioDataset(args,train_df, train =True)
        valid_dataset = AudioDataset(args,valid_df, validation=True)
        test_dataset = None

    data_collactor = CustomDataCollator(args.model_name)  

    return (train_dataset, valid_dataset, test_dataset, data_collactor)


class AudioDataset(Dataset):
    def __init__(self, args, df, train = False, validation = False, test = False):
        self.args = args
        self.df = df
        self.train = train
        self.validation = validation
        self.test = test
        
        if self.train:
            split = "train"
            self.augmentation = AudioAugmentation(self.args, train = True)
        elif self.validation:
            split = "valid"
            self.augmentation = AudioAugmentation(self.args, train = False)
        elif self.test:
            split = "test"
        else:
            raise ValueError("At least one of train, validation, or test must be True")
        
        if self.args.model_name=="resnet101":
            self.feature_extractor = FeatureExtractor('spectrogram', self.args)
        else:
            raise ValueError("Model name is False")

        data_path = f"{self.args.data_path}/librosa/{split}.pkl"
        if self.train or self.validation:
            label_path = f"{self.args.data_path}/librosa/{split}_label.pkl"

        if os.path.exists(data_path):
            print(f"{split.capitalize()} Load existing librosa file")
            print(data_path)
            with open(data_path, 'rb') as reader:
                self.librosa = pickle.load(reader)
            if self.train or self.validation:
                with open(label_path, 'rb') as reader:
                    self.labels = pickle.load(reader)
        else:
            if train or validation:
                self.librosa, self.labels = get_librosa_feature(args, df, True)
            else:
                self.librosa = get_librosa_feature(args, df, False)
            
            if not os.path.exists(f"{self.args.data_path}/librosa"):
                os.mkdir(f"{self.args.data_path}/librosa")
            
            with open(data_path, 'wb') as writer:
                pickle.dump(self.librosa, writer)
            if train or validation:
                with open(label_path, 'wb') as writer:
                    pickle.dump(self.labels, writer)

        print(f"{split} size:", len(self.librosa))
        
    def __len__(self):
        return len(self.librosa)
    
    def __getitem__(self, idx) -> Tuple:
        #when train/validation
        if self.train or self.validation:
            data1 = self.librosa[idx]
            label1 = self.labels[idx]

            idx2 = idx
            while idx2==idx:
                idx2 = np.random.randint(0,len(self.librosa)) #create audio idx to combine
            data2 = self.librosa[idx2]
            label2 = self.labels[idx2]

            data, label, people_cnt, is_noise = self.augmentation.augmentation(data1, data2, label1, label2)
            data = self.feature_extractor.extract_features(data)
            
            return {"data" : data, "label" : label, "people_cnt": people_cnt, "is_noise": is_noise} #is_noise: int np array
        #when test
        else:
            data =self.librosa[idx]
            data = self.feature_extractor.extract_features(data)
            return {"data" : data}


class CustomDataCollator():
    def __init__(self, model_type):
        self.model_type = model_type

    def __call__(self, features):

        data = [torch.tensor(feature["data"]) for feature in features]
        data = torch.stack(data).to(torch.float32)

        if "label" in features[0].keys():  # train/valid
            label = np.array([feature["label"] for feature in features])
            label = torch.tensor(label, dtype=torch.float32)
        else:  # test
            label = None

        if "people_cnt" in features[0].keys():  # train/valid
            people_cnt = np.array([feature["people_cnt"] for feature in features])
            people_cnt = torch.tensor(people_cnt, dtype=torch.long)
        else:  # test
            people_cnt = None

        if "is_noise" in features[0].keys():  # train/valid
            is_noise = np.array([feature["is_noise"] for feature in features])
            is_noise = torch.tensor(is_noise, dtype=torch.float32)
        else:  # test
            is_noise = None

        return {"data": data, "label": label, "people_cnt": people_cnt, "is_noise":is_noise}