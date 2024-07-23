import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold

def get_validation_index(args, 
                         train_df:pd.DataFrame) -> int:
    """
    To make every people's validation set equal, save validation index and use it
    """
    
    #original train/valid
    if args.fold_iter==-1:
        num_of_validation = int(len(train_df) * args.valid_ratio)
        validation_index_path = f'{args.data_path}/valid_indices_{num_of_validation}.csv'
        if os.path.exists(validation_index_path):
            validation_index = np.loadtxt(validation_index_path).astype(int)
        else:
            validation_index = np.random.choice(train_df.index, size=num_of_validation, replace=False).astype(int)
            np.savetxt(validation_index_path, validation_index, delimiter=',')
    #kfold train/valid
    else:
        num_of_validation = int(len(train_df) * args.valid_ratio)
        validation_index_path = f'{args.data_path}/5kfold_valid_indices_{args.fold_iter}_{num_of_validation}.csv'

        #load previous 5kfold indices
        if os.path.exists(validation_index_path):
            validation_index = np.loadtxt(validation_index_path).astype(int)
        #if not exist, make new 5kfold indices
        else:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            folds = list(kf.split(train_df))  
            
            for i in range(5):
                train_indices, validation_indices = folds[i]
                validation_index_path = f'{args.data_path}/5kfold_valid_indices_{i}_{num_of_validation}.csv'
                np.savetxt(validation_index_path, validation_indices, delimiter=',')
    
            validation_index = folds[args.fold_iter]

    
    return validation_index
