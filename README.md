<h1 align="center"><span style="display: inline-block; vertical-align: middle; padding-top: 50px;">
    <img src="./website/images/ihlab_logo.jpg" height="40px">
  </span>
  SwishFormer
</h1>
<h2 align="center">
  SwishFormer for Robust Firmness and Ripeness Recognition of Fruits using Visual Tactile Imagery
</h2>


<div align="center">
  <a href="https://scholar.google.com/citations?user=WMcSpaAAAAAJ&hl=en">Mashood M. Mohsan</a> &nbsp;•&nbsp;
  <a href="https://scholar.google.com/citations?user=6ixcL4cAAAAJ&hl=en">Basma Hasanen</a> &nbsp;•&nbsp;
  <a href="https://scholar.google.com.pk/citations?user=11mwy0YAAAAJ&hl=en">Taimur Hassan</a> &nbsp;•&nbsp;
  <a href="https://scholar.google.com/citations?hl=es&user=vPNmbjAAAAAJ">Muhayyuddin Ahmed</a> &nbsp;•&nbsp;
  <br/>
  <a href="https://scholar.google.com.pk/citations?user=G_2Xpm0AAAAJ&hl=en">Naoufel Werghi</a> &nbsp;•&nbsp;
  <a href="https://scholar.google.com/citations?user=MIqCjoIAAAAJ&hl=en">Lakmal Seneviratne</a> &nbsp;•&nbsp;
  <a href="https://scholar.google.com/citations?user=bCC3kdUAAAAJ&hl=en">Irfan Hussain</a> &nbsp;•&nbsp;
  <br/>
</div>

<h4 align="center">
  <a href="https://mashood3624.github.io/SwishFormer/"><b>Website</b></a> &nbsp;•&nbsp;
  <a href=""><b>Paper</b></a> &nbsp;•&nbsp; 
  <a href="https://kuacae-my.sharepoint.com/:u:/g/personal/mashood_mohsan_ku_ac_ae/EUY1BMvole1MtbwqDExjeY4BZDVbqGdIj-wqzuIuNmoXGQ?e=9cMVwK"><b>Dataset</b></a> &nbsp;•&nbsp; 
  <a href=""><b>Video</b></a>
</h4>

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) &nbsp;&nbsp;&nbsp;&nbsp; <img height="40" src="./website/images/Khalifa_logo.png" alt="Meta-AI" />  &nbsp;&nbsp; <img height="40" src="./website/images//KUCARS.jpg" alt="rpl" />
</div>

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg




<div align="center">
    SwishFormer uses tactile images when a robot palpates a fruit to predict frimness and ripeness.
  <img src="./website/images/overall.png"
  width="80%">
</div>



## Setup

### 1. Clone repository
```bash
git clone git@github.com:Mashood3624/SwishFormer.git
```
### 2. Download dataset & weights
Please download the dataset and weights by clicking <a href="https://kuacae-my.sharepoint.com/:u:/g/personal/mashood_mohsan_ku_ac_ae/EUY1BMvole1MtbwqDExjeY4BZDVbqGdIj-wqzuIuNmoXGQ?e=9cMVwK"><b>here</b></a>.
### 3. Folder structure
```bash
SwishFormer
├── data                                  # Folder contains palpation recording
    ├── Avocado_01                            # Palpation recording
    ├── Dataset.xlsx                          # Excel dataset consist of firmness values and ripness stages
├── features                              # Consist concatenated features csv files form ablated SwishFormer model for each fold 
    ├──All_kfolds_feat_all_data.csv           
├── folds_info                            # CSV files containing list of samples used in each fold (total 5 folds)
    ├── Fold_1_train.csv                     
├── inference                             # Random images to perform inference
    ├── 3 Digit images                     
├── weights                               # Consist weights of ablated Swishfromer & Random Forest regressor trained on each fold seperately
    ├── Exp_001_Fold1                        
    ├── random_forest_K_fold_1.joblib        
├── .py files                             # Rest of the files in this repo

```
### 4. Setup SwishFormer conda env
```bash
cd SwishFormer
conda env create -f environment.yml
conda activate SwishFormer
```

### 5. Train SwishFormer
Three steps to train from scratch: Train ablated SwishFormer, Extract concatenated features, & Train Random Forest regressor. 
```bash
python base.py
python features.py
python random_forest.py

```

### 6. Inference SwishFormer
Place any three 3 consecutive DIGIT images in inference folder and run the following.
```bash
python inference.py

```

<div align="Center">
    <h3>Fruit Sorting using SwishFormer </h3>
  <img src="./website/videos/fruit_sorting.gif"
  width="80%">
    
</div>


## Bibtex
```
@article{mohsan2025swishformer,
  author    = {Mohsan, Mashood M. and Hasanen, Basma B. and Hassan, Taimur and Din, Muhayy Ud and Werghi, Naoufel and Seneviratne, Lakmal and Hussain, Irfan},
  title     = {SwishFormer for Robust Firmness and Ripeness Recognition of Fruits using Visual Tactile Imagery},
  journal   = {postharvest biology and technology},
  year      = {2025},
}
```

## Acknowledgements

The majority of MidasTouch is licensed under MIT license, however portions of the project are available under separate license terms: MinkLoc3D is licensed under the MIT license; FCRN-DepthPrediction is licensed under the BSD 2-clause license; pytorch3d is licensed under the BSD 3-clause license. Please see the [LICENSE](LICENSE) file for more information.
