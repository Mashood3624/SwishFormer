from transformers import AutoFeatureExtractor, AutoModel, AutoConfig, logging
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from data import ClassificationDataset
from model import Regression_model
from safetensors.torch import load_file
import warnings
import contextlib
import io
import glob

logging.set_verbosity_error()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def extract_features(dataset, model, split_name, fold_name, df_all, counter):
    for i in tqdm(range(len(dataset)), desc=f"{fold_name} {split_name}"):
        sample = dataset[i]
        with torch.no_grad():
            out = model.forward(
                pixel_values_1=sample["pixel_values_1"].unsqueeze(0).to(device),
                pixel_values_2=sample["pixel_values_2"].unsqueeze(0).to(device),
                pixel_values_3=sample["pixel_values_3"].unsqueeze(0).to(device),
            )
        m = 0
        for z in out["concat"]:
            df_all["feat_"+str(m)][counter]=float(z)
            m=m+1
        df_all.at[counter, "labels"] = sample["labels"]
        df_all.at[counter, "Split"] = split_name
        df_all.at[counter, "Fold_name"] = fold_name
        counter += 1
    return counter

def model_init():
    model_config = AutoConfig.from_pretrained(
        encoder,
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    )
    
    model_config.num_encoder_blocks = 8
    model_config.padding = [2,1,1,1,1,1,1,1]
    model_config.patch_sizes = [7,3,3,3,3,3,3,3]
    model_config.strides = [4,2,2,2,2,2,2,2]
    model_config.hidden_sizes = [64,128,128,320,320,448,448,512]
    model_config.depths = [4,4,12,4,4,4,4,4]
    model = AutoModel.from_pretrained( encoder, config=model_config,ignore_mismatched_sizes=True)

    model = Regression_model(model)
    
    for i in range(len(model.backbone.encoder.block)):
        for j in range(len(model.backbone.encoder.block[i])):
            model.backbone.encoder.block[i][j].pooling.pool = nn.Hardswish()
            
    return model.to(device)

# Setup
exp_no = "001"

folds_name = [f"K_fold_{i+1}" for i in range(5)]
ck_pt = [f"./weights/Exp_"+exp_no+f"_Fold{i+1}/checkpoint-{chk}/model.safetensors" for i, chk in enumerate([970, 980, 720, 980, 980])]

encoder = "sail/poolformer_s24"
feature_extractor = AutoFeatureExtractor.from_pretrained(encoder) 

id2label = {
        0: 'Kiwi',
        1: 'Avocado'
    }

label2id = {label:id for id,label in id2label.items()}

all_trains = [
    "Fold_1_train.csv",
    "Fold_2_train.csv",
    "Fold_3_train.csv",
    "Fold_4_train.csv",
    "Fold_5_train.csv"
]

all_valids = [
    "Fold_1_valid.csv",
    "Fold_2_valid.csv",
    "Fold_3_valid.csv",
    "Fold_4_valid.csv",
    "Fold_5_valid.csv"    
]

all_tests = [
    "Fold_1_test.csv",
    "Fold_2_test.csv",
    "Fold_3_test.csv",
    "Fold_4_test.csv",
    "Fold_5_test.csv" 
]

counter = 0
for k_num, fold_n in enumerate(folds_name):
    X_train = pd.read_csv("./folds_info/"+all_trains[k_num])
    X_valid = pd.read_csv("./folds_info/"+all_valids[k_num])
    X_test = pd.read_csv("./folds_info/"+all_tests[k_num])

    X_train = X_train.reset_index()
    X_valid = X_valid.reset_index()
    X_test = X_test.reset_index()

    train_dataset = ClassificationDataset(X_train, feature_extractor)
    valid_dataset = ClassificationDataset(X_valid, feature_extractor)
    test_dataset = ClassificationDataset(X_test, feature_extractor)

    df_all = pd.DataFrame(index=range(len(train_dataset)+len(valid_dataset)+len(test_dataset)), 
                          columns=['Fold_name', 'Split', 'labels'] + [f"feat_{x}" for x in range(1536)])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()):
            model = model_init()
            model.load_state_dict(load_file(ck_pt[k_num]))
            model.eval()

    # Extract Features
    counter = extract_features(train_dataset, model, "Train", fold_n, df_all, counter)
    counter = extract_features(valid_dataset, model, "Valid", fold_n, df_all, counter)
    counter = extract_features(test_dataset, model, "Test", fold_n, df_all, counter)

    counter = 0
    df_all.to_csv(f"./features/All_kfolds_feat_fold_{fold_n}.csv", index=False)

csv_files = glob.glob("./features/All_kfolds_feat_fold_*.csv")
combined_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
combined_df.to_csv("./features/All_kfolds_feat_all_data.csv", index=False)
