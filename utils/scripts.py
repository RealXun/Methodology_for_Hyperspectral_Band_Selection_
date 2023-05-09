import os  # importing the 'os' module to work with the operating system
import numpy as np  # importing the 'numpy' library and renaming it to 'np'
from xgboost import XGBClassifier  # importing the XGBoost classifier from the 'xgboost' library
import time  # importing the 'time' module to measure time
import datetime  # importing the 'datetime' module to work with dates and times
import sys  # importing the 'sys' module for system-specific parameters and functions
import pandas as pd  # importing the 'pandas' library and renaming it to 'pd'
from sklearn.decomposition import PCA  # importing the 'PCA' class from the 'sklearn' library for PCA analysis
from sklearn.metrics import accuracy_score  # importing the 'accuracy_score' function from the 'sklearn' library for evaluating classification accuracy
from sklearn.model_selection import train_test_split  # importing the 'train_test_split' function from the 'sklearn' library for splitting data into training and testing sets

os.chdir(os.path.dirname(sys.path[0])) # This command makes the notebook the main path and can work in cascade.
main_folder = sys.path[0]
data_folder = (main_folder + "/" +"data")
raw_data = (data_folder + "/" + "raw")
logs_folder = (data_folder + "/" + "logs")

def PCA_band_removal(hsi_image, gt, test_size=0.9, random_state=42, var_percentage=0.999):
    
    # Get the current date and time and format the date as a string in the format "YYYY-MM-DD"
    now = datetime.datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    
    # get the name of the hsi_image variable
    hsi_image_name = [name for name in globals() if globals()[name] is hsi_image][0]

    # Check the name of the input hyperspectral image and set the log file name accordingly
    if hsi_image_name == 'salinas':
        hsi_image_name = 'log'
        log_file_name = f"salinas_{date_str}.txt"
    elif hsi_image_name == 'pavia_u':
        hsi_image_name = 'log'
        log_file_name = f"pavia_u_{date_str}.txt"
    elif hsi_image_name == 'pavia_c':
        hsi_image_name = 'log'
        log_file_name = f"pavia_c_{date_str}.txt"
    elif hsi_image_name == 'indian_pines':
        hsi_image_name = 'log'
        log_file_name = f"indian_pines_{date_str}.txt"
    else:
        log_file_name = f"default_{date_str}.txt"

    # Open the log file in append mode
    with open(os.path.join(logs_folder, log_file_name), 'a') as f:
        sys.stdout = f
        print('\n', flush=True)
        print('--- Log started on {} ---\n'.format(now.strftime('%Y-%m-%d %H:%M:%S')), flush=True)  

        starting_time = time.time() # start the timer to measure the function's execution time
        n_samples = hsi_image.shape[0] * hsi_image.shape[1] # get the number of samples in the image
        n_bands = hsi_image.shape[2] # get the number of bands in the image
        hsi_image_reshaped = hsi_image.reshape(n_samples, n_bands) # reshape the image into a 2D array of samples and bands
        print('Reshaping done', flush=True) 
        print('\n', flush=True) 

        X_train, X_test, y_train, y_test = train_test_split(hsi_image_reshaped, gt.reshape(-1), stratify=gt.reshape(-1), test_size=test_size, random_state=random_state) # split the reshaped image into training and testing datasets
        xgb = XGBClassifier(booster='gbtree', tree_method='gpu_hist', objective='multi:softmax', random_state=random_state) # create an XGBoost classifier
        xgb.fit(X_train, y_train) # fit the classifier to the training data
        feature_importances = xgb.feature_importances_ # get the feature importances from the trained classifier
        sorted_indices = np.argsort(feature_importances) # sort the indices of feature importances in ascending order

        hsi_image = hsi_image_reshaped.copy() # create a copy of the reshaped image for further processing
        original_hsi_image = hsi_image.copy() # create a copy of the original image for reference
        indices_deleted = [] # initialize a list to store the band indices that are deleted
        round_count = 1 # initialize a variable to keep track of the round number
        components = 0 # initialize a variable to store the number of components used in PCA
        overall_best_accuracy = 0 # initialize a variable to store the overall best accuracy
        deleted_bands_previous_round = None # initialize a variable to store the band indices deleted in the previous round

        print(f'Trying PCA with all bands ({hsi_image_reshaped.shape[1]} bands)', flush=True) 
        pca = PCA(n_components=var_percentage) # create a PCA object with the specified variance percentage
        hsi_image_limited = pca.fit_transform(hsi_image) # perform PCA on the reshaped image
        X_train, X_test, y_train, y_test = train_test_split(hsi_image_limited, gt.reshape(-1), stratify=gt.reshape(-1), test_size=test_size, random_state=random_state) # split the PCA-transformed data into training and testing datasets
        xgb = XGBClassifier(booster='gbtree', tree_method='gpu_hist', objective='multi:softmax', random_state=random_state) # create an XGBoost classifier for PCA-transformed data

        start_time_fit = time.time() # record the start time of fitting
        xgb.fit(X_train, y_train) # fit the classifier to the training data
        end_time_fit = time.time() # record the end time of fitting

        start_time_pred = time.time() # record the start time of predicting
        y_pred = xgb.predict(X_test) # make predictions on the testing data
        end_time_pred = time.time() # record the end time of predicting

        best_accuracy = accuracy_score(y_test, y_pred) # calculate the accuracy of the predictions
        starting_acc = best_accuracy # store the starting accuracy as the best accuracy
        print(f'Starting best accuracy: {best_accuracy}. Components: {hsi_image_limited.shape[1]}', flush=True) 
        print(f'XGBoost fitting time: {end_time_fit - start_time_fit} seconds', flush=True)
        print(f'XGBoost prediction time: {end_time_pred - start_time_pred} seconds', flush=True) 
        print(f'XGBoost total time: {end_time_pred - start_time_fit} seconds', flush=True) 
        print('-------------------------------------', flush=True) 
        print('\n', flush=True) 

        while best_accuracy > overall_best_accuracy:  # loop until the overall best accuracy is no longer improving
            print(f'ROUND {round_count}')  # print the current round number
            round_count +=1  # increment the round counter
            sorted_indices = [value for value in sorted_indices if value not in indices_deleted]  # remove the indices that were deleted in previous rounds
            overall_best_accuracy = best_accuracy  # update the overall best accuracy with the best accuracy from the previous round
            best_accuracy = starting_acc  # reset the best accuracy to the starting accuracy
            deleted_bands_previous_round = indices_deleted  # record the indices deleted in the previous round
            for index in sorted_indices:  # loop through the sorted indices
                print(f'Deleted band with index {index} (total bands if this one is deleted: {hsi_image.shape[1]})', flush=True)  # print the index of the deleted band and the total number of bands if this band is deleted
                indices_deleted.append(index)  # add the index of the deleted band to the list of deleted indices
                hsi_image = np.delete(hsi_image, indices_deleted, axis=1)  # remove the deleted bands from the HSI image
                pca = PCA(n_components=var_percentage)  # create a new PCA object with the specified number of components
                hsi_image_limited = pca.fit_transform(hsi_image)  # apply PCA to the HSI image
                X_train, X_test, y_train, y_test = train_test_split(hsi_image_limited, gt.reshape(-1), stratify=gt.reshape(-1), test_size=test_size, random_state=random_state)  # split the data into training and testing sets
                xgb = XGBClassifier(booster='gbtree', tree_method='gpu_hist', objective='multi:softmax', random_state=random_state)  # create a new XGBoost classifier

                start_time = time.time()  # record the start time of fitting and predicting
                xgb.fit(X_train, y_train)  # train the classifier on the training set
                fit_time = time.time() - start_time  # calculate the time taken for fitting the classifier

                start_time = time.time()  # record the start time of prediction
                y_pred = xgb.predict(X_test)  # use the classifier to predict the labels of the test set
                predict_time = time.time() - start_time  # calculate the time taken for predicting with the classifier
                
                acc = accuracy_score(y_test, y_pred)  # calculate the accuracy of the classifier
                print(f'Accuracy after deleting band nº{index}: {acc} (total deleted: {len(indices_deleted)} bands)', flush=True)  
                print(f'XGBoost fitting time: {fit_time} seconds', flush=True)
                print(f'XGBoost prediction time: {predict_time} seconds', flush=True)
                print(f'Total XGBoost fitting and prediction time: {fit_time+predict_time} seconds', flush=True)
                print('Evaluating results...', flush=True)
                print(f'Time elapsed: {time.time() - start_time} seconds', flush=True)  

                if best_accuracy > acc:  # if the accuracy is worse than the current best accuracy, undo the deletion of the current band
                    indices_deleted.remove(index)
                    print(f'The accuracy is worse. Adding back band {index} (total deleted: {len(indices_deleted)} bands)', flush=True)  
                    print(f'Current best accuracy: {best_accuracy}. Components: {components}', flush=True)  
                else:  # if the accuracy improves, update the best accuracy and record the number of components

                    components = hsi_image_limited.shape[1]  # number of components of the PCA
                    print(f'The accuracy improves (+{acc - best_accuracy})...', flush=True)  
                    print(f'Current best accuracy: {acc}. Components: {components}', flush=True)  
                    best_accuracy = acc  # update the best accuracy with the current accuracy
                hsi_image = original_hsi_image.copy()  # reset the HSI image to the original image
                print(f'Deleted bands up to this point: {indices_deleted}', flush=True)  
                print('-------------------------------------', flush=True)  

            print(f'Best accuracy this round: {best_accuracy}', flush=True)  
            print(f'Improvement over standard PCA: {best_accuracy - starting_acc}', flush=True)  
            print(f'Duration of the process: {time.time() - starting_time} seconds' , flush=True)  
            print('-------------------------------------', flush=True)  
            print('\n', flush=True)  

        print(f'Overall best accuracy: {overall_best_accuracy}', flush=True)  # print the overall best accuracy achieved
        print(f'Bands to remove before PCA: {deleted_bands_previous_round}', flush=True)  # print the bands removed before PCA


    # Restore stdout to its original state
    sys.stdout = sys.__stdout__
    
    return indices_deleted  # return the list of indices of the deleted bands