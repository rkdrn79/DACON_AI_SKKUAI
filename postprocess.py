import pandas as pd
import argparse
import numpy as np

def inv_softsigmoid(y, alpha = 1):
    return np.log(y / (1 - y)) * alpha

def softsigmoid(x, alpha = 2):
    return 1 / (1 + np.exp(-x / alpha))

def main(args):
    filenames = [
        f"submit/{args.df_name}_iter0.csv",
        f"submit/{args.df_name}_iter1.csv",
        f"submit/{args.df_name}_iter2.csv",
        f"submit/{args.df_name}_iter3.csv",
        f"submit/{args.df_name}_iter4.csv",        
    ]

    for i in range(5):
        data = pd.read_csv(filenames[i])
        if i == 0:
            fake = data['fake']
            real = data['real']
        else:
            fake += data['fake']
            real += data['real']
            
    submit = pd.read_csv(filenames[0])
    submit['fake'] = fake / 5
    submit['real'] = real / 5

    submit['fake'] = softsigmoid(inv_softsigmoid(submit['fake']), alpha=1.7)
    submit['real'] = softsigmoid(inv_softsigmoid(submit['real']), alpha=1.7)

    for name in filenames:
        people_df = pd.read_csv(f'{name}_class_preds.csv')
        submit.loc[people_df['zero'] >= 0.5, ['fake', 'real']] = 0 # zero가 0.5 이상인 경우 fake, real을 0으로 설정

    submit.to_csv(f'submit/{args.df_name}_final.csv',index=False)


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description="2024 DACON AI Track")
    parser.add_argument('--df_name', type=str, default="Resnet101_5kfold", help='submit name')
    args = parser.parse_args()
    main(args)