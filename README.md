# DMF_NILM
Folder containing all(hopefully) files used for creation of paper titled "A Comparison of Machine Learning Algorithms to Non-Intrusive Load Monitoring"

`environment.yml` contains the necessary libraries required to run the code. Note that different versions of CUDA, CUDNN, Tensorflow, (and maybe some others) would likely be required depending on your machine.
`main_script_sep_aug.py` is the main file that applies all ML models to the pre-made data from `data_generation/applying_augmentation.ipynb`(you will have to do this first, see_below) and is designed to be run as as part of an array job on SLURM.
`metrics_presenting.ipynb` creates all the graphics from the reports and model history generated after testing


`data_generation` folder contains all the files used to load in and process the PLAID and WHITED datasets
Link to Each
  PLAID: https://figshare.com/articles/dataset/PLAID_2017/11605215
  WHITED: https://www.cs.cit.tum.de/dis/resources/whited/

Within `data_generation`:
  `applying_augmentation.ipynb` loads in the datasets separately, applies the augmentation functions in `data_augmentation_funcs.py`, and saves the extracted and augmented data as numpy arrays.
