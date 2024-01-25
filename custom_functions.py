import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
import joblib


def grid_search_models(train_and_val_data, privileged_groups=[{'DIS': 1}], unprivileged_groups=[{'DIS': 2}], reweight=False):
    # Set up loop for hyperparameter tuning/exploration (C, regularisation strength)
    results = pd.DataFrame(columns=['C', 'Solver','Mean accuracy', 'Mean EOD'])

    best_accuracy = 0 # initialize best accuracy
    best_eod = 1 # initialize best EOD

    for C in [ 10**i for i in range(-8, 1, 1)]: # loop through different C values
        for solver in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']: # loop through different solvers
            print("Training model with C =", C, "and solver =", solver)
            
            # Initialise model
            model = LogisticRegression(solver=solver, C=C, random_state=0)

            # Set up loop for different train-val splits
            accuracy_list = [] # initialize list to store accuracy values
            eod_list = [] # initialize list to store EOD values

            for seed_index in [10*i for i in range(5)]:
                # Split train-val set into train and val sets
                train_data, val_data = train_and_val_data.split([0.8], shuffle=True, seed=seed_index)

                if reweight:
                    # Transform the original dataset via reweighing
                    RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
                    train_data = RW.fit_transform(train_data)
                    
                # Normalize the train and val datasets
                scale_orig = StandardScaler()
                x_train = scale_orig.fit_transform(train_data.features)
                y_train = train_data.labels.ravel()
                x_val = scale_orig.transform(val_data.features)
                y_val = val_data.labels.ravel()

                # Model training and prediction
                model.fit(x_train,y_train)
                predictions = model.predict(x_val)
                val_pred = val_data.copy()
                val_pred.labels = predictions

                # Metrics
                metrics = ClassificationMetric(val_data, val_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
                eod = (metrics.equal_opportunity_difference()) # Equal opportunity difference
                
                # Save model if it has the best accuracy or EOD
                if metrics.accuracy() > best_accuracy:
                    joblib.dump(model, 'std_model_accuracy.joblib')
                    best_accuracy = metrics.accuracy()
                if abs(metrics.equal_opportunity_difference()) < best_eod:
                    joblib.dump(model, 'std_model_eod.joblib')
                    best_eod = abs(metrics.equal_opportunity_difference())
                
                # Store accuracy and EOD values
                accuracy_list.append(metrics.accuracy())
                eod_list.append(eod)
            
            # Calculate mean accuracy and EOD values and append to lists
            mean_accuracy = np.mean(accuracy_list)
            mean_eod = np.mean(eod_list)
            
            # Append results to dataframe
            new_result  = pd.DataFrame([[C, solver, mean_accuracy, mean_eod]], columns=['C', 'Solver','Mean accuracy', 'Mean EOD'])
            results = pd.concat([results, new_result], ignore_index=True)

    return results