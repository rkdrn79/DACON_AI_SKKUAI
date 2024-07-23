import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import warnings
warnings.filterwarnings("ignore")

from transformers import TrainingArguments
import random
import numpy as np
import wandb
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

from src.dataset import get_dataset
from src.model import get_model
from src.trainer import BasicTrainer
from src.utils.metric import auc_brier_ece_custom
from arguments import get_arguments

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#used when check validation metric
def compute_metrics(eval_preds):

    metric = dict()

    #---------- ACC 구하기 -----------------#
    pred = eval_preds.predictions.copy()
    pred[pred>0.5] = 1
    pred[pred!=1] = 0
    label = eval_preds.label_ids.copy()
    label[label>0.5] = 1
    label[label!=1] = 0
    accuracy = np.sum(pred==label) / (len(pred)*2)
    metric['accuracy'] = accuracy
    #--------------------------------------#
    
    additional_metric = auc_brier_ece_custom(eval_preds.label_ids,eval_preds.predictions)
    metric.update(additional_metric)
    
    print("===== Validation Example ===============")
    sample_idx = np.random.randint(0,len(eval_preds.label_ids),5)
    label = eval_preds.label_ids.copy()
    pred = eval_preds.predictions.copy()
    for idx in sample_idx:
        print(f"sample {idx}")
        print(f"answer: {label[idx]}")
        print(f"pred: {pred[idx]}\n")
    print("=======================================")
    
    return metric



def main(args):
    
    model = get_model(args)
    train_ds, valid_ds, _, data_collator = get_dataset(args)
    
    #wandb.init(project='2024_SW_Festival_SubmitCode', name=f'{args.save_dir}')
    
    training_args = TrainingArguments(
        output_dir=f"./model/{args.save_dir}",
        evaluation_strategy='epoch',
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size, 
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=30,
        load_best_model_at_end=True,
        metric_for_best_model="combined_score",
        save_total_limit=8,
        remove_unused_columns=False,
        #report_to='wandb',
        dataloader_num_workers=8,
        greater_is_better=False
    )

    trainer = BasicTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print(args)
    
    trainer.train()


if __name__=="__main__":
    
    args = get_arguments()
    main(args)