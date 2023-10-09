# Getting Started

XXX is an open source, dataset-specific contrastive learning-based framework that enables two-step fast molecular pattern visualization followed by accurate protein localization without the need of any human annotation. During the exploration step, it learns an embedding space for 3D macromolecules such that similar structures are grouped together while dissimilar ones are separated. The embedding space is then projected into 2D and 3D which allows easy identification of the distribution of macromolecular structures across an entire dataset. During the refinement step, examples of proteins identified during the exploration step are selected and XXX learns to localize these proteins with high accuracy. 


Each step can be used separately. If to use refinement step only (tomogram particle detection), ground truth training coordinates need to be provided. Typically, around 200 coordinates from several tomograms are needed to ensure performance. Training coordinates can be obtained either through exploration step or manually. 

## Installation

The code was tested on CentOS Stream version 8.0, with [Anaconda](https://www.anaconda.com/download) Python 3.8 and [PyTorch]((http://pytorch.org/)) version 1.11.0 and cu 10.2. NVIDIA GPUs with 32GB RAMs are needed for training and inference can be performed on either GPU or CPU. 

After install Anaconda: 

0. [Optional but recommended] create a new conda environment. 

    ```
    conda create --name TomoPick python=3.8
    ```
    
    And activate the environment.
    
    ```
    conda activate TomoPick
    ```

1. Clone this repo:

    ```
    TomoPick_ROOT=/path/to/clone/TomoPick 
    ```

    ```
    git clone https://github.com/ $TomoPick_ROOT
    ```

2. Install the requirements

    ```
    pip install -r requirements.txt
    ```
3. Install TomoPick package and dependencies (If shows error, try go to one level above): 

    ```
    pip install -e cet_pick
    ```
## Folder structure

```

├── data # data folder that stores all training data, can rename it if needed
│   ├── sample_train_explore_img.txt
│   ├── sample_train_refine_img.txt
│   ├── sample_train_refine_coord.txt
├── datasets # datasets fodler that contains all dataloading, sampling related code 
│   ├── dataset_factory.py # file that contains all dataset factory and sampling factory 
│   ├── tomo_*.py # data factory for different modes
│   ├── particle_*.py # sampling factory for different modes
├── trains # data fodler that stores all model training modules 
├── models # model folder that contains all model architectures for different modes 
├── utils # utils folder that contains all util functions 
├── colormap # folder that contains colormaps for 2D visualization plot display  
└── DCNv2 # folder that contains deformable convolution related operations 
├── opts.py # arguments for training
├── main.py # training file for refinement module 
├── simsiam_main.py # training file for cellular content exploration module 
├── simsiam_test_hm_3d.py # inference file for cellular content exploration module  
├── test.py # inference file for refinement/particle detection module 
├── interactive_to_training_coords.py # convert output from interactive session to training coordinates file for refinement
├── plot_2d.py # plot 2D visualization plot, generate colors used for 3D tomogram visualization, and parquet file for interactive session  
├── phoenix_visualization.py # launch interactive session 

```





<!-- For full documentation visit [mkdocs.org](https://www.mkdocs.org). -->

<!-- ## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
 -->