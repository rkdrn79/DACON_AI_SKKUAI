# 2024 DACON AI Track Submit Code
## Poster

## Google Drive Link
- [Google Drive](https://drive.google.com/drive/folders/1l0yqcwItt6ERBk_6O3Aeon3Ptr1chGDS?usp=drive_link)

## OpenSource Model
### 사용한 Pretrained 오픈소스 모델 및 출처/논문 자료는 다음과 같습니다.
- Resnet101 - Deep Residual Learning for Image Recognition [Reference](https://arxiv.org/abs/1512.03385)
- AudioLDM2 - AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining [Reference](https://arxiv.org/abs/2308.05734)
```
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
@article{audioldm2-2024taslp,
  author={Liu, Haohe and Yuan, Yi and Liu, Xubo and Mei, Xinhao and Kong, Qiuqiang and Tian, Qiao and Wang, Yuping and Wang, Wenwu and Wang, Yuxuan and Plumbley, Mark D.},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={AudioLDM 2: Learning Holistic Audio Generation With Self-Supervised Pretraining}, 
  year={2024},
  volume={32},
  pages={2871-2883},
  doi={10.1109/TASLP.2024.3399607}
}
```
## Noise Extraction
### 저희 팀은 2가지 노이즈 추출 방법을 사용하였습니다.
### 1. 오픈소스 모델인 AudioLDM2을 통해 자체적으로 노이즈를 생성
### 2. Unlabeled_data에서 노이즈 추출
### 노이즈 추출 방법이 외부 데이터가 아님을 검증하기 위해, 다음과 같은 ipynb 폴더 역시 추가로 제출하였습니다.
- noise_gan_train.ipynb                         -- AudioLDM2로부터 노이즈를 생성하는 코드입니다. (train_noise 폴더 속 데이터가 이에 해당합니다.)
- noise_gan_val.ipynb                           -- AudioLDM2로부터 노이즈를 생성하는 코드입니다. (train_noise 폴더 속 데이터가 이에 해당합니다.)
- unlabeled_noise_substract.ipynb               -- unlabel data에서 수작업으로 노이즈를 추출하는 코드입니다. (unlabeled_noise_only 및 valid_nosie 폴더 속 데이터가 이에 해당합니다.)
### 최종적으로 만들어진 노이즈들은 구글 드라이브에 업로드되어 있습니다.

## Folder Structure
### 다음과 같은 폴더 구조를 가져야 합니다. 
### 구글 드라이브로부터 data 폴더 및 model 폴더를 다운로드하여 다음과 같이 위치해주세요.
- 해당하는 데이터는 다음과 같습니다.
- data/5kfold_valid_indices... csv 파일
- data/train_noise 폴더 및 데이터
- data/valid_noise 폴더 및 데이터
- data/unlabeled_noise_only 폴더 및 데이터
- model/best_model 폴더 및 5개의 checkpoint model weight
- 2개의 빈 폴더인 model/reproduce 와 data/librosa까지 다운로드하였는지 확인해 주세요.

### 또한, DACON에서 제공한 데이터도 다음과 같이 data 폴더 내부에 위치해 주세요. 해당하는 데이터는 다음과 같습니다.
- test 폴더 및 데이터
- test.csv
- train 폴더 및 데이터
- train.csv
- unlabeled_data 폴더 및 데이터
- sample_submission.csv
### data 폴더의 .pkl 파일 및 submit 폴더 내부의 제출 파일은 train/inference 과정에서 생성됩니다.
### 최종 파일은 submit 폴더의 (Inferece만 진행 시) Resnet101_5kfold_final.csv 또는 (Reproduce 진행 시) Reproduce_Resnet101_5kfold.csv 파일 입니다.
- Inference, reproduce 과정에서 필요하지는 않지만 해당 csv 파일 역시 구글 드라이브에 올려 두었습니다.
```
.
├── arguments.py
├── data
│   ├── 5kfold_valid_indices_0_11087.csv        -- 5Kfold 각각의 validation index를 저장하는 csv 파일(이하 동일)
│   ├── 5kfold_valid_indices_1_11087.csv
│   ├── 5kfold_valid_indices_2_11087.csv
│   ├── 5kfold_valid_indices_3_11087.csv
│   ├── 5kfold_valid_indices_4_11087.csv
│   ├── librosa                                 -- librosa 파일 내부 .pkl 파일들은 코드 실행 시 자동으로 생성됩니다. 
│   │   ├── test.pkl
│   │   ├── train_label.pkl
│   │   ├── train.pkl
│   │   ├── valid_label.pkl
│   │   └── valid.pkl
│   ├── sample_submission.csv                   -- DACON에서 제공한 sample_submission.csv
│   ├── test                                    -- DACON에서 제공한 test folder
│   ├── test.csv                                -- DACON에서 제공한 test.csv
│   ├── train                                   -- DACON에서 제공한 train folder
│   ├── train.csv                               -- DACON에서 제공한 train.csv
│   ├── train_noise                             -- Diffusion으로 생성한 noise folder
│   ├── unlabeled_data                          -- DACON에서 제공한 unlabeled_data folder
│   ├── valid_noise                             -- Unlabel data에서 추출한 전처리 이후 noise folder
│   └── unlabeled_noise_only                    -- Unlabel data에서 추출한 전처리 이전 noise folder
│
├── inference.py
├── model
│   ├── best_model                              -- 리더보드 모델을 저장한 폴더, 5개의 5kfold folder로 구성
│   │   ├── Resnet101_5kfold_iter0
│   │   │   └── checkpoint-2079
│   │   │       ├── model.safetensors
│   │   │       ├── optimizer.pt
│   │   │       ├── rng_state.pth
│   │   │       ├── scheduler.pt
│   │   │       ├── trainer_state.json
│   │   │       └── training_args.bin
│   │   ├── Resnet101_5kfold_iter1
│   │   │   └── checkpoint-2079
│   │   │       ├── model.safetensors
│   │   │       ├── optimizer.pt
│   │   │       ├── rng_state.pth
│   │   │       ├── scheduler.pt
│   │   │       ├── trainer_state.json
│   │   │       └── training_args.bin
│   │   ├── Resnet101_5kfold_iter2
│   │   │   └── checkpoint-2079
│   │   │       ├── model.safetensors
│   │   │       ├── optimizer.pt
│   │   │       ├── rng_state.pth
│   │   │       ├── scheduler.pt
│   │   │       ├── trainer_state.json
│   │   │       └── training_args.bin
│   │   ├── Resnet101_5kfold_iter3
│   │   │   └── checkpoint-2079
│   │   │       ├── model.safetensors
│   │   │       ├── optimizer.pt
│   │   │       ├── rng_state.pth
│   │   │       ├── scheduler.pt
│   │   │       ├── trainer_state.json
│   │   │       └── training_args.bin
│   │   └── Resnet101_5kfold_iter4
│   │       └── checkpoint-2079
│   │           ├── model.safetensors
│   │           ├── optimizer.pt
│   │           ├── rng_state.pth
│   │           ├── scheduler.pt
│   │           ├── trainer_state.json
│   │           └── training_args.bin
│   └── reproduce                               -- Reproduce 진행 시 해당 폴더에 모델이 저장됩니다.
├── postprocess.py
├── README.md
├── requirements.txt
├── src
│   ├── dataset.py
│   ├── model_factory
│   │   ├── deepcnn.py
│   │   ├── __init__.py
│   │   └── __pycache__
│   ├── model.py
│   ├── preprocessing_factory
│   │   ├── augmentation.py
│   │   ├── feature_extractor.py
│   │   ├── get_librosa.py
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   └── vaildation_index.py
│   ├── trainer.py
│   └── utils
│       ├── __init__.py
│       ├── losses.py
│       ├── metric.py
│       └── __pycache__
├── submit
│   ├── Resnet101_5kfold_final.csv                      -- 최종적으로 생성되는 파일입니다. 
│   ├── Resnet101_5kfold_iter0.csv                      -- 5kfold inference 과정에서 생성되는 파일입니다.
│   ├── Resnet101_5kfold_iter0.csv_class_preds.csv      -- 5kfold inference 과정에서 생성되는 파일입니다.(이하동일)
│   ├── Resnet101_5kfold_iter1.csv
│   ├── Resnet101_5kfold_iter1.csv_class_preds.csv
│   ├── Resnet101_5kfold_iter2.csv
│   ├── Resnet101_5kfold_iter2.csv_class_preds.csv
│   ├── Resnet101_5kfold_iter3.csv
│   ├── Resnet101_5kfold_iter3.csv_class_preds.csv
│   ├── Resnet101_5kfold_iter4.csv
│   └── Resnet101_5kfold_iter4.csv_class_preds.csv
└── train.py
```

## Conda Environment
### requirements.txt 파일에 필요한 라이브러리들을 기재해 두었습니다.
### 대회에서 사용한 개발 환경은 다음과 같습니다.
- Python 3.8.10
- CPU : Intel(R) Xeon(R) Gold 5317 CPU @ 3.00GHz 12코어
- GPU : A5000 24GB X 2
- RAM : 250GB
- Cuda : 11.7
- pip : 24.0
### 주의 사항
- 데이터 로드를 빠르게 하기 위해, 모든 데이터를 사전에 로드 후 .pkl 파일로 저장합니다.
- 따라서, 80GB 이상의 디스크 공간이 필요합니다.

## Inference from checkpoint
### Step 1
- 최종 제출본은 5-Fold 방법의 앙상블로 이루어져 있습니다. 
- 우선, 5개의 각 모델에 대하여 inference를 수행합니다.
```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
        --is_inference true \
        --data_path "./data" \
        --model_name "resnet101" \
        --save_dir "best_model/Resnet101_5kfold_iter0" \
        --weight_name "checkpoint-2079" \
        --submit_name "Resnet101_5kfold_iter0.csv"
```
```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
        --is_inference true \
        --data_path "./data" \
        --model_name "resnet101" \
        --save_dir "best_model/Resnet101_5kfold_iter1" \
        --weight_name "checkpoint-2079" \
        --submit_name "Resnet101_5kfold_iter1.csv"
```
```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
        --is_inference true \
        --data_path "./data" \
        --model_name "resnet101" \
        --save_dir "best_model/Resnet101_5kfold_iter2" \
        --weight_name "checkpoint-2079" \
        --submit_name "Resnet101_5kfold_iter2.csv"
```
```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
        --is_inference true \
        --data_path "./data" \
        --model_name "resnet101" \
        --save_dir "best_model/Resnet101_5kfold_iter3" \
        --weight_name "checkpoint-2079" \
        --submit_name "Resnet101_5kfold_iter3.csv"
```
```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
        --is_inference true \
        --data_path "./data" \
        --model_name "resnet101" \
        --save_dir "best_model/Resnet101_5kfold_iter4" \
        --weight_name "checkpoint-2079" \
        --submit_name "Resnet101_5kfold_iter4.csv"
```
### Step 2
- 이후, 다음 코드를 수행하여 후처리를 진행합니다.
- 사용한 후처리는 다음과 같습니다
    - '5 fold mean'
    - '5 fold에서 한번이라도 사람이 0명 등장한다고 예측한 부분의 real/fake 확률값을 모두 0으로 변환'
    - '예측 확률값에 soft sigmoid 적용'
```bash
python postprocess.py --df_name "Resnet101_5kfold"
```

## How to Reproduce
### Step 1
- 모델 weight 재현을 위해 5Kfold 모델 5개 각각을 학습합니다. 다음 명령어를 사용하여 5개의 모델 각각을 학습합니다.
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
        --SR 32000 \
        --data_path "./data" \
        --valid_ratio 0.2 \
        --model_name "resnet101" \
        --save_dir "reproduce/Resnet101_5kfold_iter0" \
        --learning_rate 3e-5 \
        --per_device_train_batch_size 32 \
        --gradient_accumulation_steps 4 \
        --per_device_eval_batch_size 32 \
        --num_train_epochs 10 \
        --warmup_ratio 0.1 \
        --fold_iter 0
```
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
        --SR 32000 \
        --data_path "./data" \
        --valid_ratio 0.2 \
        --model_name "resnet101" \
        --save_dir "reproduce/Resnet101_5kfold_iter1" \
        --learning_rate 3e-5 \
        --per_device_train_batch_size 32 \
        --gradient_accumulation_steps 4 \
        --per_device_eval_batch_size 32 \
        --num_train_epochs 10 \
        --warmup_ratio 0.1 \
        --fold_iter 1
```
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
        --SR 32000 \
        --data_path "./data" \
        --valid_ratio 0.2 \
        --model_name "resnet101" \
        --save_dir "reproduce/Resnet101_5kfold_iter2" \
        --learning_rate 3e-5 \
        --per_device_train_batch_size 32 \
        --gradient_accumulation_steps 4 \
        --per_device_eval_batch_size 32 \
        --num_train_epochs 10 \
        --warmup_ratio 0.1 \
        --fold_iter 2
```
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
        --SR 32000 \
        --data_path "./data" \
        --valid_ratio 0.2 \
        --model_name "resnet101" \
        --save_dir "reproduce/Resnet101_5kfold_iter3" \
        --learning_rate 3e-5 \
        --per_device_train_batch_size 32 \
        --gradient_accumulation_steps 4 \
        --per_device_eval_batch_size 32 \
        --num_train_epochs 10 \
        --warmup_ratio 0.1 \
        --fold_iter 3
```
```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
        --SR 32000 \
        --data_path "./data" \
        --valid_ratio 0.2 \
        --model_name "resnet101" \
        --save_dir "reproduce/Resnet101_5kfold_iter4" \
        --learning_rate 3e-5 \
        --per_device_train_batch_size 32 \
        --gradient_accumulation_steps 4 \
        --per_device_eval_batch_size 32 \
        --num_train_epochs 10 \
        --warmup_ratio 0.1 \
        --fold_iter 4
```
### Step 2
- 마찬가지로, 5개의 모델 각각을 inference합니다.
```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
        --is_inference true \
        --data_path "./data" \
        --model_name "resnet101" \
        --save_dir "reproduce/Resnet101_5kfold_iter0" \
        --weight_name "checkpoint-2079" \
        --submit_name "Reproduce_Resnet101_5kfold_iter0.csv"
```
```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
        --is_inference true \
        --data_path "./data" \
        --model_name "resnet101" \
        --save_dir "reproduce/Resnet101_5kfold_iter1" \
        --weight_name "checkpoint-2079" \
        --submit_name "Reproduce_Resnet101_5kfold_iter1.csv"
```
```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
        --is_inference true \
        --data_path "./data" \
        --model_name "resnet101" \
        --save_dir "reproduce/Resnet101_5kfold_iter2" \
        --weight_name "checkpoint-2079" \
        --submit_name "Reproduce_Resnet101_5kfold_iter2.csv"
```
```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
        --is_inference true \
        --data_path "./data" \
        --model_name "resnet101" \
        --save_dir "reproduce/Resnet101_5kfold_iter3" \
        --weight_name "checkpoint-2079" \
        --submit_name "Reproduce_Resnet101_5kfold_iter3.csv"
```
```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
        --is_inference true \
        --data_path "./data" \
        --model_name "resnet101" \
        --save_dir "reproduce/Resnet101_5kfold_iter4" \
        --weight_name "checkpoint-2079" \
        --submit_name "Reproduce_Resnet101_5kfold_iter4.csv"
```
### Step 3
- 이후, 다음 코드를 수행하여 후처리를 진행합니다.
- 사용한 후처리는 다음과 같습니다
    - '5 fold mean'
    - '5 fold에서 한번이라도 사람이 0명 등장한다고 예측한 부분의 real/fake 확률값을 모두 0으로 변환'
    - '예측 확률값에 soft sigmoid 적용'
```bash
python postprocess.py --df_name "Reproduce_Resnet101_5kfold"
```
