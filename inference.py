import torch.nn as nn
import joblib                                         
from transformers import AutoModel, AutoFeatureExtractor ,AutoConfig, logging
from safetensors.torch import load_file
from pathlib import Path
from PIL import Image
import warnings
from model import Regression_model
import warnings
import contextlib
import io

logging.set_verbosity_error()

warnings.filterwarnings("ignore")

def model_init():
    model_config = AutoConfig.from_pretrained(
        encoder,
        num_labels=1
    )
    
    model_config.num_encoder_blocks = 8
    model_config.padding = [2,1,1,1,1,1,1,1]
    model_config.patch_sizes = [7,3,3,3,3,3,3,3]
    model_config.strides = [4,2,2,2,2,2,2,2]
    model_config.hidden_sizes = [64,128,128,320,320,448,448,512]
    model_config.depths = [4,4,12,4,4,4,4,4]
    model = AutoModel.from_pretrained( encoder, config=model_config,ignore_mismatched_sizes=True)

    model = Regression_model(model,loaded_rf)
    
    for i in range(len(model.backbone.encoder.block)):
        for j in range(len(model.backbone.encoder.block[i])):
            model.backbone.encoder.block[i][j].pooling.pool = nn.Hardswish()     
    return model


encoder = "sail/poolformer_s24"
ck_pt = "./weights/Exp_001_Fold5/checkpoint-980/model.safetensors"

feature_extractor = AutoFeatureExtractor.from_pretrained(encoder) 
loaded_rf = joblib.load("./weights/random_forest_K_fold_5.joblib")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with contextlib.redirect_stdout(io.StringIO()):
        model = model_init()
        model.load_state_dict(load_file(ck_pt))


images = Path("./inference/").glob("*")
image_strings = [str(p) for p in images]
image_strings.sort()

predictions = []

image_path_1 = image_strings[0]
image_path_2 = image_strings[1]    
image_path_3 = image_strings[2]

image_1 = Image.open(image_path_1).convert("RGB")
image_2 = Image.open(image_path_2).convert("RGB")
image_3 = Image.open(image_path_3).convert("RGB")

pixel_values_1 = feature_extractor(image_1, return_tensors="pt").pixel_values
pixel_values_2 = feature_extractor(image_2, return_tensors="pt").pixel_values
pixel_values_3 = feature_extractor(image_3, return_tensors="pt").pixel_values

predictions.append(model(pixel_values_1,pixel_values_2,pixel_values_3)["logits"][0])

print("Firmness "+ str(predictions[0].item()))

