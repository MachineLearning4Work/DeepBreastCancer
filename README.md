# DeepBreastCancer - (To be completed soon) 

#Full Research is here : 

https://ieeexplore.ieee.org/document/8621307


Breast Cancer Histopathological Image Classification: A Deep Learning Approach

Breast cancer remains the most common type of cancer and the leading cause of cancer-induced mortality
among women with 2.4 million new cases diagnosed and 523, 000 deaths per year. Historically, a diagnosis has been
initially performed using clinical screening followed by histopathological analysis. Automated classification of cancers
using histopathological images is a challenging task of accurate detection of tumor sub-types. This process could be facilitated by machine learning approaches, which may be more reliable and economical compared to conventional method.


To prove this principle, we applied fine-tuned pre-trained deep neural networks and first attempted to discriminate be-
tween different cancer types. Using 6, 402 tissue microarrays (TMAs) samples, models including the ResNet V1 50 pre-
trained model correctly predicted 99.8% of the four cancer types including breast, bladder, lung, and lymphoma. Then,
for classification of breast cancer sub-types, this approach was applied to 7, 909 images of 82 patients from the BreakHis
database. ResNet V1 152 classified benign and malignant breast cancers with an accuracy of 98.7%. In addition, ResNet V1
50 and ResNet V1 152 categorized either benign- (adeno-sis, fibroadenoma, phyllodes tumor, and tubular adenoma) or
malignant- (ductal carcinoma, lobular carcinoma, mucinous carcinoma, and papillary carcinoma) sub-types with 94.8% and
96.4% accuracy, respectively. The confusion matrices revealed high sensitivity values of 1, 0.995 and 0.993 for cancer types, as well as malignant- and benign sub-types respectively. The areas under the curve (AUC) scores were 0.996, 0.973 and 0.996 for
cancer types, malignant and benign sub-types, respectively. One of the most significant and striking result to emerge from the
data analysis is negligible false positive (FP) and false negative (FN). The optimum results, as shown in Tables, indicate that FP is between 0 and 4 while FN is between 0 and 8 in which test data including 800, 900, 809, 1000 for given four classes

# Requirements : 

Python 3+  

Platform Linux Ubuntu 16.04

NVIDIA GPU

Tensorflow-gpu 1.7+ and its dependencies  ( https://www.tensorflow.org/install/gpu) 


# How to run : 
Extract all files and keep that folders name as it is.


Gentle reminder:  4.TFslim_fine_tune.zip should be Extracted and kept as a folder with its own contents 


guidline.pdf  is a very early basic guide (Version 1 is incomplete, But ship it anyway)


# Support By : 

This work has been supported in part by a  start-up  fund  from  Weill  Cornell Medicine  and the Iranian National Elite  Foundation   and   grants   provided   by   Royan   Institute.


# Citation : 

If you find this code/ research useful in your research, please consider citing:

Plain Text : 
M. Jannesari et al., "Breast Cancer Histopathological Image Classification: A Deep Learning Approach," 2018 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), Madrid, Spain, 2018, pp. 2405-2412.

BibTex:

@INPROCEEDINGS{BreastCancerBIBM2018,
author={M. Jannesari and M. Habibzadeh and H. Aboulkheyr and P. Khosravi and O. Elemento and M. Totonchi and I. Hajirasouliha},
booktitle={2018 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
title={Breast Cancer Histopathological Image Classification: A Deep Learning Approach},
year={2018},
volume={},
number={},
pages={2405-2412},
keywords={Breast cancer;Deep learning;Tumors;Pathology;Training;Convolutional Neural Network;Deep learning;Digital Pathology Imaging;Breast cancer},
doi={10.1109/BIBM.2018.8621307},
ISSN={},
month={Dec},}



# Contact Info : 



Mahboobeh Jannesari (mahboobeh.jannesary@gmail.com)

Mehdi Habibzadeh  (me_habi@encs.concordia.ca) 

Hamidreza AB ES (hamidrezaab@gmail.com)

Mehdi Totonchi (totonchimehdi@gmail.com)

Iman Hajirasouliha (imh2003@med.cornell.edu)

Olivier Elemento (ole2001@med.cornell.edu)
