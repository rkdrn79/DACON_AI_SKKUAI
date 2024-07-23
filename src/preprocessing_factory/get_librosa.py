import os
from tqdm import tqdm
import pandas as pd
import librosa
import numpy as np

def get_librosa_feature(args,
                     df:pd.DataFrame,
                     is_train:bool):
    features = []
    labels = []
    
    for _, row in tqdm(df.iterrows(), desc="making librosa_feature..."):
        #load every audio file in dataframe
        
        cur_path = os.path.join(args.data_path, row['path'])
        y, sr = librosa.load(cur_path, sr=args.SR)
        
        features.append(y)

        if is_train:
            label = row['label']
            label_vector = np.zeros(2, dtype=np.float32) 
            label_vector[0 if label == 'fake' else 1] = 1
            labels.append(label_vector)
        
    if is_train:
        return features, labels
    return features