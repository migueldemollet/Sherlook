<img src="https://github.com/migueldemollet/real-or-fake-image-machine-learning/blob/main/resource/sherlook-logo.png" align="right" width="300" alt="header pic"/>


# SHERLOOK
SHERLOOK is an advanced Deep Learning model that provides users with a powerful tool to detect fake news by identifying images as either original or modified, while also indicating potential areas of alteration.

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
This repository showcases the work and results of implementing a deep learning model using TensorFlow. The primary objective of this project is to detect whether an image has been modified, either by software or by an AI, with the aim of combating the spread of fake news. By leveraging advanced techniques in deep learning, the model is trained to analyze image features and accurately classify images as either authentic or modified. The repository provides a comprehensive overview of the model architecture, training process, evaluation metrics, and the implementation code. The ultimate goal is to contribute to the development of tools that can aid in verifying the authenticity of images, thereby helping to mitigate the impact of fake news in various domains.

If you would like to delve deeper into the details of this project, you can refer to the accompanying paper, which can be accessed at the following link: [Paper](https://github.com/migueldemollet/real-or-fake-image-machine-learning/blob/main/doc/Final_Report_of_Bachelor_Thesis.pdf). The paper provides comprehensive information about the methodology, experimental setup, results, and analysis, offering a more in-depth understanding of the project's contributions and findings.

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
Various experiments have been conducted with different architectures and image preprocessing techniques. The first part of the text represents the architecture used, while the last part represents the image preprocessing technique. E stands for Error Level Analysis, W stands for Wavelet, and YUV stands for utilizing the YUV color space.

| Modelo      | Épocas | Tiempo por Época | Accuracy | Loss | Precisión | Recall | AUC | PRC | F1-Score |
|-------------|--------|-----------------|----------|------|-----------|--------|-----|-----|----------|
| ENB1\_v2\_E | 13     | 99s             | 0.93     | 0.21 | 0.95      | 0.94   | 0.98| 0.98| 0.93     |
| ENB3\_E     | 14     | 126s            | 0.92     | 0.20 | 0.95      | 0.90   | 0.98| 0.98| 0.92     |
| XC\_E       | 12     | 147s            | 0.90     | 0.55 | 0.90      | 0.93   | 0.93| 0.94| 0.92     |
| MN\_E       | 12     | 64s             | 0.91     | 0.22 | 0.99      | 0.84   | 0.98| 0.99| 0.91     |
| MN\_YUV     | 33     | 45s             | 0.92     | 0.23 | 0.90      | 0.92   | 0.97| 0.97| 0.91     |
| ENVB2\_E    | 31     | 104s            | 0.89     | 0.30 | 0.84      | 0.98   | 0.97| 0.97| 0.91     |
| ENB1\_E     | 31     | 100s            | 0.89     | 0.31 | 0.85      | 0.84   | 0.97| 0.97| 0.90     |
| XC\_YUV     | 20     | 130s            | 0.82     | 0.83 | 0.78      | 0.98   | 0.91| 0.90| 0.87     |
| V16\_E      | 15     | 25s             | 0.87     | 0.38 | 0.80      | 0.86   | 0.93| 0.87| 0.83     |
| ENV2B1\_E   | 29     | 60s             | 0.78     | 0.68 | 0.71      | 1      | 0.95| 0.94| 0.83     |
| ENB1\_YUV   | 18     | 104s            | 0.63     | 0.69 | 0.62      | 1      | 0.49| 0.58| 0.77     |
| R50\_E      | 7      | 32s             | 0.83     | 0.46 | 0.81      | 0.72   | 0.93| 0.88| 0.76     |
| XC\_W       | 11     | 158s            | 0.62     | 0.61 | 0.62      | 1      | 0.50| 0.62| 0.76     |
| V16\_W      | 20     | 108s            | 0.62     | 0.67 | 0.62      | 1      | 0.50| 0.62| 0.76     |
| ENB1\_W     | 16     | 125s            | 0.61     | 0.67 | 0.91      | 1      | 0.49| 0.60| 0.76     |
| V16\_YUV    | 14     | 89s             | 0.63     | 0.65 | 0.65      | 0.90   | 0.61| 0.70| 0.75     |
| Scrath\_W   | 15     | 42s             | 0.60     | 0.68 | 0.56      | 1      | 0.50| 0.60| 0.75     |
| MN\_W       | 20     | 67s             | 0.60     | 0.68 | 0.60      | 1      | 0.50| 0.60| 0.75     |
| R50\_W      | 14     | 90s             | 0.60     | 0.68 | 0.60      | 1      | 0.50| 0.60| 0.75     |
| Scrath\_E   | 12     | 43s             | 0.74     | 1.64 | 0.94      | 0.32   | 0.85| 0.82| 0.48     |

If you want test the others models you can download the models [here](https://drive.google.com/file/d/1-rf4MdVF-zJcjfLtEm8QyWMkPcemDtOs/view?usp=sharing)

Visual results have been obtained to provide a visual representation of the potential modifications made. The best result is showcased, highlighting the specific modification that has been implemented. These visual results serve as a demonstration of how the modifications impact the overall output.

![grad-cam](https://github.com/migueldemollet/real-or-fake-image-machine-learning/blob/main/result/grad-cam.png?raw=true)

The confusion matrix will be presented to further analyze and understand the test results. The confusion matrix provides a detailed breakdown of the model's predictions, showing the number of true positive, true negative, false positive, and false negative instances. It offers valuable information on the model's performance, allowing for a deeper understanding of its accuracy and potential areas of improvement.

![confusion matrix](https://github.com/migueldemollet/real-or-fake-image-machine-learning/blob/main/result/confusion_matrix.png?raw=true)

The model training process was completed in approximately 25 minutes. The training and validation metrics are provided to evaluate the performance of the model. These metrics offer insights into how well the model was trained and how it performed on both the training and validation datasets.

![metrics](https://github.com/migueldemollet/real-or-fake-image-machine-learning/blob/main/result/metrics.png?raw=true)
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
│   |   ├── cat.jpg
│   |   ├── me_x_3.jpg
│   |   ├── edited_by_ia.jpg
│   ├── gitignore
├── model
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
│   ├── poster.png
├── src
│   ├── analisys.ipynb
│   ├── model_custom.ipynb
│   ├── models_ela_custom.ipynb
│   ├── models_ela.ipynb
│   ├── models_wavelet.ipynb
│   ├── models_yuv_custom.ipynb
├── result
│   ├── confusion_matrix.png
│   ├── grad-cam.png
│   ├── metrics.png
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

> python3 src/main.py

# Built With 🛠️

vscode - The code editor used

# License 📄
This project is under the MIT License - see the [LICENSE](https://github.com/migueldemollet/real-or-fake-image-machine-learning/blob/main/LICENSE) file for details

# How to contribute 🤝
If you want to contribute to this project, you create a pull request. All contributions are welcome.

# Support 🤝
- **Jordi Serra Raiz** - Tutor of the project - [Jordi Serra Raiz](https://www.linkedin.com/in/jordiserraruiz/)
- **Laia Guerreo Candela** - Provider of AI-generated modified images and logo design.- [Laia Guerreo Candela](https://www.linkedin.com/in/laiaguerrerocandela/)

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
