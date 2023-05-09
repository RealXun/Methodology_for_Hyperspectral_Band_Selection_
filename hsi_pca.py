import os  # importing the 'os' module to work with the operating system
import numpy as np  # importing the 'numpy' library and renaming it to 'np'
import scipy  # importing the 'scipy' library for scientific computing
from xgboost import XGBClassifier  # importing the XGBoost classifier from the 'xgboost' library
import time  # importing the 'time' module to measure time
import datetime  # importing the 'datetime' module to work with dates and times
import sys  # importing the 'sys' module for system-specific parameters and functions
import subprocess  # importing the 'subprocess' module to spawn new processes
import pandas as pd  # importing the 'pandas' library and renaming it to 'pd'
from sklearn.decomposition import PCA  # importing the 'PCA' class from the 'sklearn' library for PCA analysis
from sklearn.metrics import accuracy_score  # importing the 'accuracy_score' function from the 'sklearn' library for evaluating classification accuracy
from sklearn.model_selection import train_test_split  # importing the 'train_test_split' function from the 'sklearn' library for splitting data into training and testing sets

from utils import scripts
os.chdir(os.path.dirname(sys.path[0])) # This command makes the notebook the main path and can work in cascade.
main_folder = sys.path[0]
data_folder = (main_folder + "/" +"data")
raw_data = (data_folder + "/" + "raw")
logs_folder = (data_folder + "/" + "logs")
saved_results_folder = (data_folder + "/" + "saved_results")
indian_pines_folder = (saved_results_folder + "/" + "indian_pines")
pavia_center_folder = (saved_results_folder + "/" + "pavia_center")
pavia_university_folder = (saved_results_folder + "/" + "pavia_university")
salinas_folder = (saved_results_folder + "/" + "salinas")

pavia_u = scipy.io.loadmat(raw_data + '/' + 'PaviaU.mat')['paviaU']
pavia_u_gt = scipy.io.loadmat(raw_data + "/" + "PaviaU_gt.mat")['paviaU_gt']

pavia_c = scipy.io.loadmat(raw_data + '/' + 'Pavia.mat')['pavia']
pavia_c_gt = scipy.io.loadmat(raw_data + "/" + "Pavia_gt.mat")['pavia_gt']

salinas = scipy.io.loadmat(raw_data + '/' + 'Salinas.mat')['salinas']
salinas_gt = scipy.io.loadmat(raw_data + '/' + 'Salinas_gt.mat')['salinas_gt']

indian_pines = scipy.io.loadmat(raw_data + '/' + 'Indian_pines.mat')['indian_pines']
indian_pines_gt = scipy.io.loadmat(raw_data + '/' + 'Indian_pines_gt.mat')['indian_pines_gt']

# define menu options
menu_options = {
    1: "Indian Pines",
    2: "Pavia University",
    3: "Pavia Centre",
    4: "Salinas"
}

# display menu options
print("Select an option:")
for option in menu_options:
    print(f"{option}: {menu_options[option]}")

# prompt for user input
while True:
    try:
        user_input = int(input("\nEnter an option number: "))
        if user_input not in menu_options:
            print("Invalid option selected. Please try again.")
            continue
        break
    except ValueError:
        print("Invalid input. Please enter a number.")

# select code based on user input
if user_input == 1:
    print("\n\nYou have selected Indian Pines dataset.")
    data = "Indian Pines"
elif user_input == 2:
    print("\n\nYou have selected Pavia University dataset.")
    data = "Pavia University"
elif user_input == 3:
    print("\n\nYou have selected Pavia Centre dataset.")
    data = "Pavia Centre"
elif user_input == 4:
    print("\n\nYou have selected Salinas dataset.")
    data = "Salinas"

# run code based on user input
if user_input == 1:
    print("\n\nYou have selected Indian Pines dataset ")
    print("\n\nThe code is already running. You can check the log file in logs folder to see the status")
    indices_deleted = scripts.PCA_band_removal(indian_pines, indian_pines_gt)
elif user_input == 2:
    print("\n\nYou have selected Pavia University dataset ")
    print("\n\nThe code is already running. You can check the log file in logs folder to see the status")
    indices_deleted = scripts.PCA_band_removal(pavia_u, pavia_u_gt)   
elif user_input == 3:
    print("\n\nYou have selected Pavia Centre dataset ")
    print("\n\nThe code is already running. You can check the log file in logs folder to see the status")
    indices_deleted = scripts.PCA_band_removal(pavia_c, pavia_c_gt)
elif user_input == 4:
    print("\n\nYou have selected Salinas dataset ")
    print("\n\nThe code is already running. You can check the log file in logs folder to see the status")
    indices_deleted = scripts.PCA_band_removal(salinas, salinas_gt)

