# Deep Learning for Image Tampering Detection
This project aims to develop a deep learning model that can detect modified images and distinguish them from original images.

# Table of Contents 📖
   * [What is this?](#what-is-this-🤔)
   * [Demo](#demo-📺)
   * [Dataset](#dataset-💾)
   * [Results](#results-📊)
   * [Project Structure](#project-structure-📁)
   * [Requirements](#requirements-📋)
   * [How to use](#how-to-use-🚀)
   * [Built With](#built-with-🛠️)
   * [License](#license-📄)
   * [How to contribute](#How-to-contribute-🤝)
   * [Citing](#citing-📜)
   * [Support](#support-🤝)
   * [Authors](#authors-✒️)
   * [Bibliography](#bibliography-📚)

# What is this? 🤔

# Demo 📺
[![Watch the video](https://img.youtube.com/vi/TBBw0aDBIbg/maxresdefault.jpg)](https://youtu.be/TBBw0aDBIbg)

# Dataset 💾

The dataset used in this project is the [Casia dataset](https://www.kaggle.com/datasets/sophatvathana/casia-dataset), which contains 12,614 images. The images are divided into two folders: Au (original images) and Tp (modified images).

Alternatively, you can download it from [here](https://1drv.ms/f/s!ApMviAlZmRwE6gLHz13fOw0uT04H?e=IQRgMS) where you will have the exact same dataset as the one used in this project, with some minor modifications.

The dataset includes different categories of images, such as:

* ani: (animal)
* arc: (architecture), 
* art: (art), 
* cha: (characters), 
* nat: (nature), 
* pla: (plants), 
* sec: (sections), 
* txt: (texture)

```
├── dataset
│   ├── Au
│   |   ├── Au_ani_00001.jpg
│   |   ├── Au_ani_00002.jpg
│   |   ├── ... 
│   ├── Tp
│   |   ├── Tp_D_CND_M_N_ani00018_sec00096_00138.jpg
│   |   ├── Tp_D_CND_M_N_art00076_art00077_10289.jpg
│   |   ├── ...
```

To create the dataset with tampered images, it is important to note that we have two subcategories within the main categories:

* D: Different
* S: Same

This is because images can be modified in two ways:

* Different: The image is modified using another image.
* Same: The image is modified using the same image.

# Results 📊

# Project Structure 📁
All the code is located in the src folder. The dataset is located in the dataset folder. The doc folder contains the final report of the project. The requirements.txt file contains all the required libraries to run the code. The gitignore file contains the files that are not uploaded to the repository. The README.md file is the file you are currently reading.
```
├── dataset
│   ├── Au
│   |   ├── Au_ani_00001.jpg
│   |   ├── Au_ani_00002.jpg
│   |   ├── ... 
│   ├── Tp
│   |   ├── Tp_D_CND_M_N_ani00018_sec00096_00138.jpg
│   |   ├── Tp_D_CND_M_N_art00076_art00077_10289.jpg
│   |   ├── ...
│   ├── test
│   |   ├── me_x_3.jpg
│   |   ├── edited_by_ia.jpg
│   ├── gitignore
├── model
│   |   ├── checkpoints
│   |   ├── logs
│   |   ├── custom_models
│   |   |   ├── efficientnetB3
│   |   |   |   ├── model_arquitecture.json
│   |   |   |   ├── model_weights.h5
│   |   |   ├── mobilenet
│   |   |   |   ├── model_arquitecture.json
│   |   |   |   ├── model_weights.h5
│   |   ├── ela_models
│   |   |   ├── detect_manipulated_images_model_scratch.h5
│   |   |   ├── ...
│   |   ├── wavelet_models
│   |   |   ├── detect_manipulated_images_model_scratch.h5
│   |   |   ├── ...
│   |   ├── yuv_models
│   |   |   ├── detect_manipulated_images_model_efficientNetB1.h5
│   |   |   ├── ...
│   |   ├── gitignore
├── doc
│   ├── Final_Report_of_Bachelor_Thesis.pdf
│   ├── Gantt_diagram.xlsx
├── src
│   ├── analisys.ipynb
│   ├── model_custom.ipynb
│   ├── models_ela_custom.ipynb
│   ├── models_ela.ipynb
│   ├── models_wavelet.ipynb
│   ├── models_yuv_custom.ipynb
├── result
│   ├── ela_models
│   |   ├── confusion_matrix
│   |   |   ├── confusion_matrix.png
│   |   |   ├── ...
│   |   ├── heatmap
│   |   |   ├── heatmap.png
│   |   |   ├── ...
│   |   ├── metrics
│   |   |   ├── metrics.png
│   |   |   ├── ...
│   ├── wavelet_models
│   |   ├── confusion_matrix
│   |   |   ├── confusion_matrix.png
│   |   |   ├── ...
│   |   ├── metrics
│   |   |   ├── metrics.png
│   |   |   ├── ...
├── gitignore
├── LICENSE
├── README.md
├── requirements.txt
```

# Requirements 📋
* Python 3.9
* All the required libraries are in the requirements.txt file
    * opencv-python
    * numpy
    * matplotlib
    * Pillow
    * Pandas
    * kaggle
    * tensorflow
    * scikit-learn
    * PyWavelets
    * keras-tuner

If you don't have some of these libraries, you can install them manually or by running the following command:
    
        pip install -r requirements.txt

# How to use 🚀
1. Clone this repo.

> git clone https://github.com/migueldemollet/real-or-fake-image-machine-learning.git

2. Go to the directory.

> cd real-or-fake-image-machine-learning

3. Install the required libraries.
 
using pip :

> pip install -r requirements.txt

using conda :

> conda install --file requirements.txt

4. Run the code.

> 

# Built With 🛠️

vscode - The code editor used

# License 📄
This project is under the MIT License - see the [LICENSE](https://github.com/migueldemollet/real-or-fake-image-machine-learning/blob/main/LICENSE) file for details

# How to contribute 🤝

# Citing 📜

# Support 🤝

# Authors ✒️
* **Miguel del Arco** - [migueldemollet](https://github.com/migueldemollet)

# Bibliography 📚
1. Adoble. Adoble Analytics, [Link](https://acortar.link/2Qtak9), 2012.

2. Raúl Álvarez. Adobe, el creador de Photoshop, está desarrollando software para detectar imágenes manipuladas... con Photoshop, [Link](https://acortar.link/PeLl6m), 2018.

3. Sheng-Yu Wang, Oliver Wang, Andrew Owens, Richard Zhang, Alexei A. Efros. Detecting Photoshopped Faces by Scripting Photoshop. ICCV, 2019.

4. Thanh Thi Nguyen, Quoc Viet Hung Nguyen, Dung Tien Nguyen, Duc Thanh Nguyen, Thien Huynh-The, Saeid Nahavandi, Thanh Tam Nguyen, Quoc-Viet Pham, Cuong M. Nguyen. Deep Learning for Deepfakes Creation and Detection: A Surveyl, *arXiv:1909.11573*, 2022.

5. Andreas Rössler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies, Matthias Nießner. FaceForensics++: Learning to Detect Manipulated Facial Images, *arXiv:1901.08971*, 2019.

6. NPHAT SOVATHANA. casia dataset v2, [Link](https://www.kaggle.com/datasets/sophatvathana/casia-dataset), 2018.

7. NPHAT SOVATHANA. casia dataset v1, [Link](https://www.kaggle.com/datasets/sophatvathana/casia-dataset), 2018.

8. MarsAnalysisProject. Image Forensics, [Link](https://forensics.map-base.info/report_2/index_en.shtml), 2016.

9. Koushik Chandrasekaran. 2D-Discrete Wavelet Transformation and its applications in Digital Image Processing using MATLAB, [Link](https://acortar.link/cq8jPp), 2021.

10. Wikipedia. YUV, [Link](https://en.wikipedia.org/wiki/YUV), 2004.

11. Jason Brownlee. Use Early Stopping to Halt the Training of Neural Networks At the Right Time, [Link](https://acortar.link/w8QGLe), 2020.

12. Xue Ying. An Overview of Overfitting and its Solutions, *10.1088/1742-6596/1168/2/022022*, 2019.

13. B. Chen. Early Stopping in Practice: an example with Keras and TensorFlow 2.0, [Link](https://acortar.link/ccPyUl), 2020.

14. Tokio School. Analizamos qué es y para qué se usa el Transfer Learning en el Deep Learning, [Link](https://www.tokioschool.com/noticias/transfer-learning/), 2022.

15. DataScientest. ¿Qué es el método Grad-CAM?, [Link](https://datascientest.com/es/que-es-el-metodo-grad-cam), 2022.

16. fchollet. Grad-CAM class activation visualization, [Link](https://keras.io/examples/vision/grad_cam/), 2020.

17. Mingxing Tan, Quoc V. Le. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks, *arXiv:1905.11946*, 2019.

18. Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, Mark Chen. Hierarchical Text-Conditional Image Generation with CLIP Latents, *arXiv:2204.06125*, 2022.

19. Jonas Oppenlaender. The Creativity of Text-to-Image Generation, *arXiv:2206.02904*, 2022.