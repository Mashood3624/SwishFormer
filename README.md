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
  <a href=""><b>Dataset</b></a> &nbsp;•&nbsp; 
  <a href=""><b>Video</b></a>
</h4>

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) &nbsp;&nbsp;&nbsp;&nbsp; <img height="40" src="./website/images/Khalifa_logo.png" alt="Meta-AI" />  &nbsp;&nbsp; <img height="40" src="./website/images//KUCARS.jpg" alt="rpl" />
</div>

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg


MidasTouch performs online global localization of a vision-based touch sensor on an object surface during sliding interactions.  For details and further results, refer to our <a href="https://suddhu.github.io/midastouch-tactile/">website</a> and <a href="https://openreview.net/forum?id=JWROnOf4w-K">paper</a>.

<div align="center">
  <img src="./website/images/overall.png"
  width="80%">
</div>



## Setup

### 1. Clone repository
```bash
git clone git@github.com:facebookresearch/MidasTouch.git
git submodule update --init --recursive
```
### 2. Download dataset & weights

### 3. Folder structure
```bash
SwishFormer
├── data                                  # Folder contains palpation recording
    ├── Avocado_01                            # Palpation recording
    ├── Dataset.xlsx                          # Excel dataset consist of firmness values and ripness stages
├── features                              # Consist concatenated features csv files form ablated SwishFormer model for each fold 
    ├──All_kfolds_feat_all_data.csv           
├── folds_info                            # CSV 
    ├── Fold_1_train.csv                      # b
├── inference                             # b
    ├── 3 Digit images                        # b
├── weights                               # b
    ├── Exp_001_Fold1                         # b
    ├── random_forest_K_fold_1.joblib         # b
├── .py files                             # b

```
### 4. Setup SwishFormer conda env
```bash
cd SwishFormer
conda env create -f environment.yml
conda activate SwishFormer
```

### 5. Train SwishFormer
Run interactive filtering experiments with our YCB-Slide data from both the simulated and real-world tactile interactions. 



### 6. Inference SwishFormer
Run interactive filtering experiments with our YCB-Slide data from both the simulated and real-world tactile interactions. 

<div align="left">
  <img src="./website/videos/fruit_sorting.gif"
  width="80%">
    
</div>


## Bibtex
```
@inproceedings{suresh2022midastouch,
    title={{M}idas{T}ouch: {M}onte-{C}arlo inference over distributions across sliding touch},
    author={Suresh, Sudharshan and Si, Zilin and Anderson, Stuart and Kaess, Michael and Mukadam, Mustafa},
    booktitle = {Proc. Conf. on Robot Learning, CoRL},
    address = {Auckland, NZ},
    month = dec,
    year = {2022}
}
```

## Acknowledgements

The majority of MidasTouch is licensed under MIT license, however portions of the project are available under separate license terms: MinkLoc3D is licensed under the MIT license; FCRN-DepthPrediction is licensed under the BSD 2-clause license; pytorch3d is licensed under the BSD 3-clause license. Please see the [LICENSE](LICENSE) file for more information.
