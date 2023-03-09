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

# Dataset 💾

The dataset used in this project is the [Casia dataset](https://www.kaggle.com/datasets/sophatvathana/casia-dataset), which contains 12,614 images. The images are divided into two folders: Au (original images) and Tp (modified images).

Alternatively, you can download it from [here]() where you will have the exact same dataset as the one used in this project, with some minor modifications.

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
│   ├── gitignore
├── doc
│   ├── Final_Report_of_Bachelor_Thesis.pdf
│   ├── Gantt_diagram.xlsx
├── src
│   ├── notebook.ipynb
├── gitignore
├── LICENSE
├── README.md
├── requirements.txt
```

# Requirements 📋
* Python 3.9 or later
* All the required libraries are in the requirements.txt file
    * opencv-python
    * numpy
    * matplotlib
    * Pillow
    * Pandas
    * kaggle

If you don't have some of these libraries, you can install them manually or by running the following command:
    
        pip install -r requirements.txt

# How to use 🚀
1. Clone this repo.

> git clone https://github.com/migueldemollet/real-or-fake-image-machine-learning.git

2. Go to the directory where the code is located.

> cd real-or-fake-image-machine-learning/src

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