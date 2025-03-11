from torch.utils.data import Dataset
from PIL import Image
import torch

class ClassificationDataset(Dataset):
    def __init__(self, df,feature_extractor):
        self.df = df
        self.feature_extractor = feature_extractor

    def __len__(self):
        return self.df.shape[0]-2

    def __getitem__(self, idx):
        image_path_1 = self.df["Image_1"][idx]
        image_path_2 = self.df["Image_2"][idx+1]
        image_path_3 = self.df["Image_3"][idx+2]
        
        #print(image_path_1,image_path_2,image_path_3)
        if "Kiwi" == self.df["labels"][idx]:
            labels = 0
        elif "Avocado" == self.df["labels"][idx]:
            labels = 1
            
        ripeness = self.df["ripeness"][idx+2]
        
        image_1 = Image.open(image_path_1).convert("RGB")
        image_2 = Image.open(image_path_2).convert("RGB")
        image_3 = Image.open(image_path_3).convert("RGB")
        
        pixel_values_1 = self.feature_extractor(image_1, return_tensors="pt").pixel_values
        pixel_values_2 = self.feature_extractor(image_2, return_tensors="pt").pixel_values
        pixel_values_3 = self.feature_extractor(image_3, return_tensors="pt").pixel_values
        
        y = float(ripeness)
        
        encoding = {
            "pixel_values_1": pixel_values_1.squeeze(),
            "pixel_values_2": pixel_values_2.squeeze(),
            "pixel_values_3": pixel_values_3.squeeze(),
            "labels":y,
            
        }
        return encoding
    
