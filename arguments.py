import argparse


def get_arguments():
    
    parser = argparse.ArgumentParser(description="2024 DACON AI Track")
    
    #=============== parser with data =====================#
    parser.add_argument('--SR', type=int, default=32000, help='Sampling Ratio')
    parser.add_argument('--data_path', type=str, default='./data', help='data path')
    parser.add_argument('--valid_ratio', type=float, default=0.2, help='validation data ratio')
    #=================================================================#
    
    #================= parser with model ===========================#
    parser.add_argument('--model_name', type=str, default='MLP', help='custom model name')
    parser.add_argument('--save_dir', type=str, default='MLP_Baseline', help='custom model name')
    parser.add_argument('--weight_name', type=str,default='checkpoint-3460',help ='model weight name')
    parser.add_argument('--submit_name', type=str,default='baseline_submit.csv',help='submit name')
    #=================================================================#

    #================= parser with train  ===========================#    
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--per_device_train_batch_size', type=int, default=32, help='Per device train batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=32, help='Per device eval batch size')
    parser.add_argument('--num_train_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')

    #================= parser with augmentation =========================#
    parser.add_argument('--noise_prob', type=float, default=0.8, help='noise apply ratio')
    parser.add_argument('--reduce_prob', type=float, default=0.3, help='reduce sound apply ratio')
    parser.add_argument('--two_people_prob', type=float, default=0.5, help='two audio sum ratio')
    parser.add_argument('--zero_people_prob', type=float, default=0.05, help='zero audio sum ratio')
    #=================================================================#

    #================= Other Arguments ================================#
    parser.add_argument('--is_inference', type=bool, default=False, help='use only when inference')
    parser.add_argument('--is_embedding', type=bool, default=False, help='use only when embedding')
    parser.add_argument('--fold_iter', type=int, default=-1, help='use only when kfold')
    #==================================================================#
    
    args = parser.parse_args()
    
    return args