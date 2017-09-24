# BRATS 2017

## Brain Tumor Image Segmentation Challenge 2017

This is an approach for brain tumor segmentation based on a multi-path CNN.
_______________________________________________________________________________________________________________________

## Instructions on how to run CNN training:  
	python train.py -config config2.json -o output-directory-path

In the configuration file (config2.json) db_path should be replaced by the path to the BRATS database  
BRATS database structure:  
- Brats17TestingData  
- Brats17TrainingData  
  - HGG  
  - LGG  
- Brats17ValidationData

output-directory-path - directory path where the meta data, model and segmentation results will be placed
_______________________________________________________________________________________________________________________
## Hardware specification:  
- GPU: nVidia's GeForce GTX 980 Ti (6 GB)  
- CPU: Intel Core i7-6700K @ 4.00 GHz (32 GB)
_______________________________________________________________________________________________________________________
## Python version: 2.7.12  
## Python libraries:  
- tensorflow '1.2.1'  
- numpy '1.13.1'  
- scipy '0.17.0'  
- natsort '1.5.1'  
- nibabel '2.1.0'  
- json '2.0.9'
_______________________________________________________________________________________________________________________
## Note:  
Following scripts and classes will be added / updated:  
- class for post-processing  
- validation tool-chain  
- testing tool-chain
_______________________________________________________________________________________________________________________
## References:

[1] Menze BH, Jakab A, Bauer S, Kalpathy-Cramer J, Farahani K, Kirby J, Burren Y, Porz N, Slotboom J, Wiest R, Lanczi L, Gerstner E, Weber MA, Arbel T, Avants BB, Ayache N, Buendia P, Collins DL, Cordier N, Corso JJ, Criminisi A, Das T, Delingette H, Demiralp Ç, Durst CR, Dojat M, Doyle S, Festa J, Forbes F, Geremia E, Glocker B, Golland P, Guo X, Hamamci A, Iftekharuddin KM, Jena R, John NM, Konukoglu E, Lashkari D, Mariz JA, Meier R, Pereira S, Precup D, Price SJ, Raviv TR, Reza SM, Ryan M, Sarikaya D, Schwartz L, Shin HC, Shotton J, Silva CA, Sousa N, Subbanna NK, Szekely G, Taylor TJ, Thomas OM, Tustison NJ, Unal G, Vasseur F, Wintermark M, Ye DH, Zhao L, Zhao B, Zikic D, Prastawa M, Reyes M, Van Leemput K. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015)

[2] Bakas S, Akbari H, Sotiras A, Bilello M, Rozycki M, Kirby JS, Freymann JB, Farahani K, Davatzikos C. "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, (2017) [In Press]

[3] Bakas S, Akbari H, Sotiras A, Bilello M, Rozycki M, Kirby J, Freymann J, Farahani K, Davatzikos C. "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-GBM collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.KLXWJJ1Q

[4] Bakas S, Akbari H, Sotiras A, Bilello M, Rozycki M, Kirby J, Freymann J, Farahani K, Davatzikos C. "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-LGG collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.GJQ7R0EF
_______________________________________________________________________________________________________________________
