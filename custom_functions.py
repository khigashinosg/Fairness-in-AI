import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
import joblib

# Function to perform grid search for the best models, exports the best models
def grid_search_models(train_and_val_data, privileged_groups=[{'DIS': 1}], unprivileged_groups=[{'DIS': 2}], reweight=False, custom_criterion_style='sum_of_squares'):
    print("reweight =", reweight)
    
    # Set up loop for hyperparameter tuning/exploration (C, regularisation strength)
    results = pd.DataFrame(columns=['C', 'Solver','Mean accuracy', 'Mean EOD', 'Mean custom criterion'])

    best_accuracy = 0 # initialize best accuracy
    best_eod = 1 # initialize best EOD
    best_custom_criterion = 0 # initialize custom criterion

    for C in [ 10**i for i in range(-6, 1, 1)]: # loop through different C values
        for solver in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']: # loop through different solvers
            print("Training model with C =", C, "and solver =", solver)
            
            # Initialise model
            model = LogisticRegression(solver=solver, C=C, random_state=0)

            # Set up loop for different train-val splits
            accuracy_list = [] # initialize list to store accuracy values
            eod_list = [] # initialize list to store EOD values
            custom_criterion_list = [] # initialize list to store custom criterion values

            for seed_index in [10*i for i in range(5)]:
                # Split train-val set into train and val sets
                train_data, val_data = train_and_val_data.split([0.8], shuffle=True, seed=seed_index)

                if reweight == True:
                    # Transform the original dataset via reweighing
                    RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
                    train_data = RW.fit_transform(train_data)
                
                # Normalize the train and val datasets
                scale_orig = StandardScaler()
                x_train = scale_orig.fit_transform(train_data.features)
                y_train = train_data.labels.ravel()
                x_val = scale_orig.transform(val_data.features)

                # Model training and prediction
                model.fit(x_train,y_train, sample_weight=train_data.instance_weights)
                predictions = model.predict(x_val)
                val_pred = val_data.copy()
                val_pred.labels = predictions

                # Metrics
                metrics = ClassificationMetric(val_data, val_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
                eod = (metrics.equal_opportunity_difference()) # Equal opportunity difference
                custom_criterion = calc_custom_criterion(metrics.accuracy(), eod, custom_criterion_style) # Custom criterion
                
                # Save model if it has the best accuracy or EOD
                if reweight: # if reweighting is used, save the model to a specific pathname
                    if metrics.accuracy() > best_accuracy:
                        joblib.dump(model, 'fair_model_accuracy.joblib')
                        best_accuracy = metrics.accuracy()
                    if abs(metrics.equal_opportunity_difference()) < best_eod:
                        joblib.dump(model, 'fair_model_eod.joblib')
                        best_eod = abs(metrics.equal_opportunity_difference())
                    if custom_criterion > best_custom_criterion:
                        joblib.dump(model, 'fair_model_cc.joblib')
                        best_custom_criterion = custom_criterion
                        
                else: # if reweighting is not used, save the model to a specific pathname
                    if metrics.accuracy() > best_accuracy:
                        joblib.dump(model, 'std_model_accuracy.joblib')
                        best_accuracy = metrics.accuracy()
                    if abs(metrics.equal_opportunity_difference()) < best_eod:
                        joblib.dump(model, 'std_model_eod.joblib')
                        best_eod = abs(metrics.equal_opportunity_difference())
                    if custom_criterion > best_custom_criterion:
                        joblib.dump(model, 'std_model_cc.joblib')
                        best_custom_criterion = custom_criterion
                
                
                # Store accuracy and EOD values
                accuracy_list.append(metrics.accuracy())
                eod_list.append(eod)
                custom_criterion_list.append(custom_criterion)
            
            # Calculate mean accuracy and EOD values and append to lists
            mean_accuracy = np.mean(accuracy_list)
            mean_eod = np.mean(eod_list)
            mean_custom_criterion = np.mean(custom_criterion_list)
            
            # Append results to dataframe
            new_result  = pd.DataFrame([[C, solver, mean_accuracy, mean_eod, mean_custom_criterion]], columns=['C', 'Solver','Mean accuracy', 'Mean EOD', 'Mean custom criterion'])
            results = pd.concat([results, new_result], ignore_index=True)

    return results



# Function to find the best results from the already performed grid search
def find_best_results(results):
    # Find highest accuracy, lowest EOD, and highest custom criterion
    highest_accuracy = results['Mean accuracy'].max()
    lowest_eod = results['Mean EOD'].min() 

    # Find lowest EOD that is not 0
    lowest_nonzero_eod = results.loc[results['Mean EOD'] != 0]['Mean EOD'].abs().min()
    
    # Find highest custom criterion (Task 3)
    highest_custom_criterion = results['Mean custom criterion'].max()


    # Find the corresponding C and solver values
    best_accuracy = results.loc[results['Mean accuracy'] == highest_accuracy]
    best_eod = results.loc[results['Mean EOD'] == lowest_eod]
    best_nonzero_eod = results.loc[(results['Mean EOD'] == lowest_nonzero_eod) | (results['Mean EOD'] == -lowest_nonzero_eod)]
    best_custom_criterion = results.loc[results['Mean custom criterion'] == highest_custom_criterion]
    
    return best_accuracy, best_eod, best_nonzero_eod, best_custom_criterion



# Function to train the best models on the whole train set and export them
# def test_models(test_data, best_accuracy_model, best_eod_model, best_cc_model, privileged_groups=[{'DIS': 1}], unprivileged_groups=[{'DIS': 2}]):
#     # Load the best model
#     model_accuracy = joblib.load(best_accuracy_model)
#     model_eod = joblib.load(best_eod_model)

#     # Normalize the test dataset
#     scale_orig = StandardScaler()
#     x_test = scale_orig.fit_transform(test_data.features)

#     # Model prediction
#     predictions_accuracy = model_accuracy.predict(x_test)
#     predictions_eod = model_eod.predict(x_test)

#     # Create test dataset with predictions
#     test_pred_accuracy = test_data.copy()
#     test_pred_accuracy.labels = predictions_accuracy
#     test_pred_eod = test_data.copy()
#     test_pred_eod.labels = predictions_eod

#     # Metrics
#     metrics_best_accuracy = ClassificationMetric(test_data, test_pred_accuracy, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
#     best_accuracy_accuracy = metrics_best_accuracy.accuracy()
#     best_accuracy_eod = metrics_best_accuracy.equal_opportunity_difference()

#     metrics_best_eod = ClassificationMetric(test_data, test_pred_eod, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
#     best_eod_accuracy = metrics_best_eod.accuracy()
#     best_eod_eod = metrics_best_eod.equal_opportunity_difference()
    
#     return best_accuracy_accuracy, best_accuracy_eod, best_eod_accuracy, best_eod_eod

def test_model(test_data, model_pathname, privileged_groups=[{'DIS': 1}], unprivileged_groups=[{'DIS': 2}]):
        # Load the best model
    model = joblib.load(model_pathname)

    # Normalize the test dataset
    scale_orig = StandardScaler()
    x_test = scale_orig.fit_transform(test_data.features)

    # Model prediction
    predictions = model.predict(x_test)

    # Create test dataset with predictions
    results = test_data.copy()
    results.labels = predictions

    # Metrics
    metrics = ClassificationMetric(test_data, results, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    accuracy = metrics.accuracy()
    eod = metrics.equal_opportunity_difference()
    custom_criterion = calc_custom_criterion(accuracy, eod, 'sum_of_squares')
    
    return accuracy, eod, custom_criterion
    

def calc_custom_criterion(accuracy, eod, style):
    # Calculate custom criterion
    fairness = 1 - abs(eod)
    
    if style == 'sum':
        custom_criterion = accuracy + fairness
    elif style == 'sum_of_squares':
        custom_criterion = accuracy**2 + fairness**2
    elif style == 'sum_of_logs':
        custom_criterion = np.log(accuracy) + np.log(fairness)
    
    return custom_criterion