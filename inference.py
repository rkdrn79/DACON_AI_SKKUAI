import torch.utils
import torch.utils.data
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from safetensors.torch import load_model

from src.dataset import get_dataset
from src.model import get_model
from arguments import get_arguments

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    
    model = get_model(args)
    load_model(model, f"./model/{args.save_dir}/{args.weight_name}/model.safetensors")
    
    _, _, test_ds, data_collator = get_dataset(args)
    
    test_dataloader = torch.utils.data.DataLoader(test_ds, 
                                                  batch_size=args.per_device_eval_batch_size,
                                                  collate_fn=data_collator,
                                                  drop_last=False,
                                                  shuffle=False)
    
    model.to(device)
    model.eval()

    predictions = []
    class_preds = []
    probs = None
    with torch.no_grad():
        for features in tqdm(iter(test_dataloader)):
            
            data = features['data'].to(device)
            probs = model(data) #[batch,2]
            probs  = probs.cpu().detach().numpy().astype(np.float32)
            
            predictions += probs[:,:2].tolist()
            class_preds += probs[:,2:].tolist()
    
    submit = pd.read_csv(f'{args.data_path}/sample_submission.csv')
    
    submit = submit.astype({ 
    'id': 'object',  
    'fake': 'float32',  
    'real': 'float32'   
    })
        
    submit.iloc[:, 1:] = predictions
    submit.head()

    print("=============== Prediction Example ===============")
    print(predictions[:10])
    print("==================================================")
    
    submit.to_csv(f'./submit/{args.submit_name}', index=False)

    class_preds_df = pd.DataFrame(class_preds, columns=['zero', 'one','two', 'is_noise'])
    class_preds_df.insert(0, 'id', submit['id']) 

    class_preds_file_name = f'./submit/{args.submit_name}_class_preds.csv'
    class_preds_df.to_csv(class_preds_file_name, index=False)

    print(f"Class predictions saved to {class_preds_file_name}")


if __name__=="__main__":
    
    args = get_arguments()
    main(args)