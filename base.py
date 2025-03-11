import pandas as pd
from evaluate import load
import torch.nn as nn
import os
from transformers import AutoModel, AutoFeatureExtractor ,AutoConfig
from transformers import TrainingArguments,Trainer
import torch
from data import ClassificationDataset
from model import Regression_model

os.environ['WANDB_DISABLED'] = 'true'

exp_no = "001"
model_folder = ["./weights/Exp_"+exp_no+f"_Fold{i}" for i in range(1,6)]
TRAIN_EPOCHS = 10
LEARNING_RATE = 0.00005
WEIGHT_DECAY = 0.01
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32

encoder = "sail/poolformer_s24"
feature_extractor = AutoFeatureExtractor.from_pretrained(encoder) 

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

def collate_fn(examples):
    pixel_values_1 = torch.stack([example["pixel_values_1"] for example in examples])
    pixel_values_2 = torch.stack([example["pixel_values_2"] for example in examples])
    pixel_values_3 = torch.stack([example["pixel_values_3"] for example in examples])
    
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values_1": pixel_values_1.squeeze(),
            "pixel_values_2": pixel_values_2.squeeze(),
            "pixel_values_3": pixel_values_3.squeeze(),
            "labels": labels}

metric_1 = load("mse")
metric_2 = load("r_squared")
metric_3 = load("mae")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    m1 = metric_1.compute(predictions=predictions, references=labels)
    m2 = metric_2.compute(predictions=predictions, references=labels)
    m3 = metric_3.compute(predictions=predictions, references=labels)
    return {"mse":m1["mse"], "r_quared":m2, "mae":m3["mae"]}


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
            
    return model


outputs_all = []
for k_num in range(len(model_folder)):
    
    X_train = pd.read_csv("./folds_info/"+all_trains[k_num])
    X_valid = pd.read_csv("./folds_info/"+all_valids[k_num])
    X_test = pd.read_csv("./folds_info/"+all_tests[k_num])

    X_train = X_train.reset_index()
    X_valid = X_valid.reset_index()
    X_test = X_test.reset_index()

    train_dataset = ClassificationDataset(
            X_train, feature_extractor
        )
    valid_dataset = ClassificationDataset(
            X_valid,feature_extractor
        )
    test_dataset = ClassificationDataset(
            X_test,feature_extractor
        )

    id2label = {
        0: 'Kiwi',
        1: 'Avocado'
    }

    label2id = {label:id for id,label in id2label.items()}

    model = model_init()
    
    training_args = TrainingArguments(
        output_dir=model_folder[k_num],
        fp16=True,
        overwrite_output_dir=True,
        evaluation_strategy = 'epoch',
        num_train_epochs=TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=VALID_BATCH_SIZE,
        logging_strategy = "epoch",
        save_strategy = "epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model = "eval_loss"

    )    

    trainer = Trainer(
        model = model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
    )
    trainer.train()

    print("Test scores of ",model_folder[k_num])
    
    results = trainer.evaluate(test_dataset)
    print(results)
    outputs_all.append(results)

t_1 = outputs_all[0]
t_2 = outputs_all[1]
t_3 = outputs_all[2]
t_4 = outputs_all[3]
t_5 = outputs_all[4]


print('%.3f'%(t_1['eval_mse']),t_1["eval_r_quared"],'%.3f'%(t_1['eval_mae']))

print('%.3f'%(t_2['eval_mse']),t_2["eval_r_quared"],'%.3f'%(t_2['eval_mae']))

print('%.3f'%(t_3['eval_mse']),t_3["eval_r_quared"],'%.3f'%(t_3['eval_mae']))

print('%.3f'%(t_4['eval_mse']),t_4["eval_r_quared"],'%.3f'%(t_4['eval_mae']))

print('%.3f'%(t_5['eval_mse']),t_5["eval_r_quared"],'%.3f'%(t_5['eval_mae']))

print("r_quared ",(t_1["eval_r_quared"]+t_2["eval_r_quared"]+t_3["eval_r_quared"]+t_4["eval_r_quared"]+t_5["eval_r_quared"])/5)
print("mse ",(t_1["eval_mse"]+t_2["eval_mse"]+t_3["eval_mse"]+t_4["eval_mse"]+t_5["eval_mse"])/5)
print("mae ",(t_1["eval_mae"]+t_2["eval_mae"]+t_3["eval_mae"]+t_4["eval_mae"]+t_5["eval_mae"])/5)