# Deep learning mask segmentation using U-net
Program developed to generate mask images of objects like rock samples and other similar objects photographed in green static background for photogrammetry reconstruction. The script encapsulated in a QT interface uses deep learning U-net semantic segmentation architecture to predict the masks from a folder chosen by the user. The script workflow separate the input images into tiles with 256x256 pixels size, predict the mask for each small tile, and unite the predicted tiles before saving the masks adding "_mask.png" to the saved files.

<img src="https://github.com/ademirmarquesjunior/deep_learning_mask_segmentation/blob/main/images/image1.png" width="500" alt="Segmented image">
<img src="https://github.com/ademirmarquesjunior/deep_learning_mask_segmentation/blob/main/images/image2.png" width="500" alt="Segmented image">


## Requirements

- Tensorflow=2.8
- Keras=2.6
- Numpy
- Pillow
- Glob
- Sys
- PyQT5


## Install

To install this software/script download and unpack the software package in any folder.

Install the required libraries individually or run the script bellow:
 
     pip install -r Requirements.txt
     

     
## Usage
 
 To run this program use:
 
    python mask_segm.py
  
Choose the the input folder with original images in jpeg format, and choose and or create the output folder where the masks will be saved. Hit "Process masks" to start the process.

<img src="https://github.com/ademirmarquesjunior/deep_learning_mask_segmentation/blob/main/images/program_interface1.png" width="500" alt="Segmented image">


* Agisoft Metashape processing requires the insertion of camera calibration information for aditional cameras besides the first. To be fixed.


## TODO

Next iterations expect to improve and incorporate:

 - Performance improvements
 - Model improvement (training)


## Credits	
This work is credited to the [Vizlab | X-Reality and GeoInformatics Lab](http://vizlab.unisinos.br/) and the following developers:	[Ademir Marques Junior](https://www.researchgate.net/profile/Ademir_Junior) and Vinicius Ferreira Sales.

## License

    MIT Licence (https://mit-license.org/)
    
## How to cite

Yet to be published.
