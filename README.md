# DECODING CLOUD MASK product from ECOSTRESS Level-2
A tutorial how to use the code ecostress_preprocessing.py to decode the Cloud Mask product from ECOSTRESS Level-2 
Documentation of ECOSTRESS L-2: https://ecostress.jpl.nasa.gov/downloads/userguides/2_ECOSTRESS_L2_UserGuide_06182019.pdf

# General description
ECOSTRESS L-2 CloudMask is a separate file that could be downloaded besides ECOSTRESS LST & Emissivity products. 
Information of how to download CloudMask could be found here: https://lpdaac.usgs.gov/products/eco2cldv001/

ECOSTRESS L-2 CloudMask is an 8-bit flag of cloud mask and cloud tests. The general information of each bit could be found in Table 7 of this file: https://ecostress.jpl.nasa.gov/downloads/userguides/2_ECOSTRESS_L2_UserGuide_06182019.pdf

The procedure to decode the 8-bit: (from left- to right)
1. Read the first bit to determine if cloud mask was calculated or not. 
2. Read the second bit to determnine if cloud was deteced or not in either of tests 1 and 2 
3. If test 1 and 2 fail, final cloud mask could be outcomes of cloud test in bits 1-3
4. If test 1 - 3 fail, using final test in bit 4 is the land/water mask 


# Instructions
Example data is uploaded in /example/ . The folder contains two files: 
ECO2LSTE.001_SDS_LST_doy2019278182133_aid0001.tif : ECOSTRESS LST
ECO2CLD.001_SDS_CloudMask_doy2019278182133_aid0001.tif : ECOSTRESS CloudMask
![image](https://user-images.githubusercontent.com/12726626/117020051-add1c600-acbb-11eb-8c26-56b958329dd9.png)

1. 
