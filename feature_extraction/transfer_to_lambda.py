
def linRegOneMetricCustom(vectors_dict, y, randShuffle=False, do_lasso=False, do_ridge=False, alpha=1.0):
    # runs simple linear regression via one-left-out
    # vectors_dict -- dictionary mapping time radius (in minutes) to features
    # y -- a numpy array with labels (self-reported metrics)
    # randShuffle -- do we shuffle the self-report labels?
    # if do_lasso, does lasso regression
    # if do_ridge, does ridge regression. Overrides do_lasso
    # alpha - this is the weighting of either lasso or ridge

    # returns a dictionary with several results:
    # scores -- dictionary mapping each time radius to list of MSEs from each one-left-out
    # preds -- dictionary mapping each time radius to a list of each one-left-out model's prediction
    # y -- returns y again for convenience
    # models -- dictionary mapping each time radius to a list of each one-left-out trained model (simple linear regression)



    # Custom implementation without cross_val_score
    scores = {}
    preds = {}
    models = {}

    if randShuffle:
        y_using = np.random.permutation(y)
    else:
        y_using = y

    for i in vectors_dict.keys():
        # Initialize appropriate model based on input flags
        model = LinearRegression()
        if do_lasso:
            if USING_CONTROLBURN:
                model = ControlBurnClassifier(alpha=alpha)
            else:
                model = Lasso(alpha=alpha)
        if do_ridge:
            model = Ridge(alpha=alpha)

        scores[i] = []
        preds[i] = np.zeros(y_using.shape)
        models_i_building = []

        # Manual leave-one-out cross-validation
        for test_index in range(vectors_dict[i].shape[0]):
            # Split the data into training and testing sets
            X_train = np.delete(vectors_dict[i], test_index, axis=0)
            y_train = np.delete(y_using, test_index, axis=0)
            X_test = vectors_dict[i][test_index:test_index+1]  # Keep the test sample in 2D
            y_test = y_using[test_index]

            # Fit the model and make predictions
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            preds[i][test_index] = y_pred

            # Compute and store the MSE for this fold
            mse = mean_squared_error(y_test, y_pred)
            scores[i].append(mse)

            models_i_building.append(model)  # Save the trained model for this fold

        models[i] = models_i_building

    return scores, preds, y_using, models

