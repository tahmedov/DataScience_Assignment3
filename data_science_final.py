#!/usr/bin/env python
# coding: utf-8


def data_ex():
    import pandas as pd
    import matplotlib.pyplot as plt

    # Load the data
    train_df = pd.read_csv("train.csv", encoding="ISO-8859-1")
    product_desc_df = pd.read_csv("product_descriptions.csv", encoding="ISO-8859-1")
    attributes_df = pd.read_csv("attributes.csv")

    # Explore train.csv

    # Number of product-query pairs
    num_pairs = len(train_df)
    print("Number of product-query pairs:", num_pairs)

    # Number of unique products
    num_products = train_df["product_uid"].nunique()
    print("Number of unique products:", num_products)

    # Two most occurring products and their frequency
    top_products = train_df["product_title"].value_counts().head(2)
    print("Two most occurring products:")
    print(top_products)

    # Descriptive statistics for relevance values
    relevance_stats = train_df["relevance"].describe()
    print("Relevance statistics:")
    print(relevance_stats)

    # Distribution of relevance values (histogram)
    plt.hist(train_df["relevance"])
    plt.title("Distribution of relevance values")
    plt.xlabel("Relevance")
    plt.ylabel("Count")
    plt.show()

    # Explore product_descriptions.csv

    # Check the number of unique product descriptions
    num_product_desc = product_desc_df["product_uid"].nunique()
    print("Number of unique product descriptions:", num_product_desc)

    # Explore attributes.csv

    # Top 5 most occurring brand names in product attributes
    top_brands = attributes_df[attributes_df["name"] == "MFG Brand Name"]["value"].value_counts().head(5)
    print("Top 5 most occurring brand names in product attributes:")
    print(top_brands)
    
data_ex()





def t_tdidf():
    ### 3 MODELS WITH TF-IDF ###
    import pandas as pd 
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import mean_squared_error
    from scipy import sparse
    import numpy as np
    from catboost import CatBoostRegressor

    # Load the data
    train_df = pd.read_csv("train.csv", encoding="ISO-8859-1")
    product_desc_df = pd.read_csv("product_descriptions.csv", encoding="ISO-8859-1")

    # Merge train_df with product_desc_df based on product_uid
    df = pd.merge(train_df, product_desc_df, how='left', on='product_uid')

    # Preprocess the text data
    df['search_term'] = df['search_term'].str.lower()
    df['product_title'] = df['product_title'].str.lower()
    df['product_description'] = df['product_description'].str.lower()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df[['search_term', 'product_title', 'product_description']], df['relevance'], test_size=0.2, random_state=42)

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the search term, product title, and product description
    X_train_search_term = vectorizer.fit_transform(X_train['search_term'])
    X_train_product_title = vectorizer.transform(X_train['product_title'])
    X_train_product_description = vectorizer.transform(X_train['product_description'])

    X_test_search_term = vectorizer.transform(X_test['search_term'])
    X_test_product_title = vectorizer.transform(X_test['product_title'])
    X_test_product_description = vectorizer.transform(X_test['product_description'])

    # Convert sparse matrices to dense arrays
    X_train_tfidf = sparse.hstack([X_train_search_term, X_train_product_title, X_train_product_description])
    X_test_tfidf = sparse.hstack([X_test_search_term, X_test_product_title, X_test_product_description])
    # Train and evaluate models
    models = [
        LinearRegression(),
        Ridge(alpha=1.0),
        CatBoostRegressor()
    ]

    for model in models:
        # Train the model
        model.fit(X_train_tfidf, y_train)

        # Predict on the testing set
        y_pred = model.predict(X_test_tfidf)

        # Evaluate the model using RMSE
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        model_name = model.__class__.__name__
        print(f"{model_name} - Root Mean Squared Error (RMSE): {rmse}")
        
t_tdidf()





def baseline():
    ### BASELINE RMSE ###
    from sklearn.linear_model import LinearRegression, Ridge
    from catboost import CatBoostRegressor
    from sklearn.metrics import mean_squared_error

    # Fit a linear regression model to the training set
    linear_regression = LinearRegression()
    linear_regression.fit(X_train_raw, y_train)

    # Fit a ridge regression model to the training set
    ridge_regression = Ridge(alpha=1.0)
    ridge_regression.fit(X_train_raw, y_train)

    # Fit a CatBoostRegressor model to the training set
    catboost_regressor = CatBoostRegressor()
    catboost_regressor.fit(X_train_raw, y_train, verbose=False)

    # Calculate the mean of the target variable in the training set
    mean_relevance = y_train.mean()

    # Predict the mean value for all instances in the testing set
    baseline_pred_linear = [mean_relevance] * len(y_test)

    # Predict using the ridge regression model
    baseline_pred_ridge = ridge_regression.predict(X_test_raw)

    # Predict using the CatBoostRegressor model
    baseline_pred_catboost = catboost_regressor.predict(X_test_raw)

    # Evaluate the baseline predictions using RMSE
    baseline_rmse_linear = mean_squared_error(y_test, baseline_pred_linear, squared=False)
    baseline_rmse_ridge = mean_squared_error(y_test, baseline_pred_ridge, squared=False)
    baseline_rmse_catboost = mean_squared_error(y_test, baseline_pred_catboost, squared=False)

    print("Baseline RMSE for Linear Regression:", baseline_rmse_linear)
    print("Baseline RMSE for Ridge Regression:", baseline_rmse_ridge)
    print("Baseline RMSE for CatBoostRegressor:", baseline_rmse_catboost)
    
baseline()





def grid_reg():
    ### Grid search hyperparameter tunning ###
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_squared_error

    # Fit a linear regression model to the training set
    linear_regression = LinearRegression()
    linear_regression.fit(X_train_raw, y_train)

    # Define the parameter grid for Ridge Regression
    param_grid = {'alpha': [0.1, 1.0, 10.0]}

    # Perform grid search to find the best alpha value for Ridge Regression
    ridge_regression = Ridge()
    grid_search = GridSearchCV(ridge_regression, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_raw, y_train)

    # Get the best alpha value from the grid search
    best_alpha = grid_search.best_params_['alpha']

    # Fit a Ridge Regression model with the best alpha value
    ridge_regression_best = Ridge(alpha=best_alpha)
    ridge_regression_best.fit(X_train_raw, y_train)

    # Calculate the mean of the target variable in the training set
    mean_relevance = y_train.mean()

    # Predict the mean value for all instances in the testing set
    baseline_pred_linear = [mean_relevance] * len(y_test)

    # Predict using the linear regression model
    baseline_pred_ridge = ridge_regression_best.predict(X_test_raw)

    # Evaluate the baseline predictions using RMSE
    baseline_rmse_linear = mean_squared_error(y_test, baseline_pred_linear, squared=False)
    baseline_rmse_ridge = mean_squared_error(y_test, baseline_pred_ridge, squared=False)

    print("GridSearch optimized RMSE for Linear Regression:", baseline_rmse_linear)
    print("GridSearch optimized RMSE for Ridge Regression:", baseline_rmse_ridge)
    
grid_reg()





def grid_cat():
    ### Grid search hyperparameter tunning ###
    from catboost import CatBoostRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import GridSearchCV

    # Define the parameter grid for CatBoostRegressor
    param_grid = {
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.1, 1],
        'iterations': [100, 200, 300]
    }

    # Perform grid search to find the best hyperparameters for CatBoostRegressor
    catboost = CatBoostRegressor()
    grid_search = GridSearchCV(catboost, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_raw, y_train)

    # Get the best hyperparameters from the grid search
    best_depth = grid_search.best_params_['depth']
    best_learning_rate = grid_search.best_params_['learning_rate']
    best_iterations = grid_search.best_params_['iterations']

    # Fit a CatBoostRegressor model with the best hyperparameters
    catboost_best = CatBoostRegressor(depth=best_depth, learning_rate=best_learning_rate, iterations=best_iterations)
    catboost_best.fit(X_train_raw, y_train)

    # Predict using the CatBoostRegressor model
    baseline_pred_catboost = catboost_best.predict(X_test_raw)

    # Evaluate the baseline prediction using RMSE
    baseline_rmse_catboost = mean_squared_error(y_test, baseline_pred_catboost, squared=False)

    print("GridSearch optimized RMSE for CatBoostRegressor:", baseline_rmse_catboost)
    
grid_cat()





def stemmed_data():
    ### RMSE WITH STEMMED DATA ###
    import pandas as pd
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import LinearRegression, Ridge
    from catboost import CatBoostRegressor
    from nltk.stem import SnowballStemmer
    from sklearn.feature_extraction.text import CountVectorizer
    from scipy.sparse import hstack

    # Load the data
    train_df = pd.read_csv("train.csv", encoding="ISO-8859-1")
    product_desc_df = pd.read_csv("product_descriptions.csv", encoding="ISO-8859-1")

    # Merge train_df with product_desc_df based on product_uid
    df = pd.merge(train_df, product_desc_df, how='left', on='product_uid')

    # Initialize Snowball Stemmer
    stemmer = SnowballStemmer('english')

    # Apply stemming to the text columns
    df['search_term'] = df['search_term'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
    df['product_title'] = df['product_title'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
    df['product_description'] = df['product_description'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
    df.to_csv("stemmed_data.csv", index=False)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df[['search_term', 'product_title', 'product_description']],
                                                        df['relevance'], test_size=0.2, random_state=42)

    # Create CountVectorizer
    vectorizer = CountVectorizer()

    # Fit and transform the search term, product title, and product description
    X_train_search_term = vectorizer.fit_transform(X_train['search_term'])
    X_train_product_title = vectorizer.transform(X_train['product_title'])
    X_train_product_description = vectorizer.transform(X_train['product_description'])

    X_test_search_term = vectorizer.transform(X_test['search_term'])
    X_test_product_title = vectorizer.transform(X_test['product_title'])
    X_test_product_description = vectorizer.transform(X_test['product_description'])

    # Concatenate the feature matrices
    X_train_stemmed = hstack([X_train_search_term, X_train_product_title, X_train_product_description])
    X_test_stemmed = hstack([X_test_search_term, X_test_product_title, X_test_product_description])

    # Train and evaluate models
    models = [
        LinearRegression(),
        Ridge(alpha=1.0),
        CatBoostRegressor()
    ]

    for model in models:
        # Train the model
        model.fit(X_train_stemmed, y_train)

        # Predict on the testing set
        y_pred = model.predict(X_test_stemmed)

        # Calculate RMSE
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        model_name = model.__class__.__name__
        print(f"{model_name} - Root Mean Squared Error (RMSE): {rmse}")
        r2 = r2_score(y_test, y_pred)
        print("R2 Score:", r2)
        
stemmed_data()





def random_reg():
    ### RandomizedSearch of the 2 best models and runtime calculation###
    import time
    from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import RandomizedSearchCV

    # Set up parameter grid for random search optimization
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    }

    # Perform random search optimization for RandomForestRegressor
    start_time = time.time()
    random_forest = RandomForestRegressor()
    random_search_rf = RandomizedSearchCV(random_forest, param_distributions=param_grid, n_iter=10, cv=5)
    random_search_rf.fit(X_train_raw, y_train)
    end_time = time.time()

    # Calculate the runtime for random search optimization
    runtime_random_search_rf = end_time - start_time

    print("Random search optimization runtime for RandomForestRegressor:", runtime_random_search_rf, "seconds")

    # Perform random search optimization for BaggingRegressor
    start_time = time.time()
    bagging = BaggingRegressor()
    random_search_bagging = RandomizedSearchCV(bagging, param_distributions=param_grid, n_iter=10, cv=5)
    random_search_bagging.fit(X_train_raw, y_train)
    end_time = time.time()

    # Calculate the runtime for random search optimization
    runtime_random_search_bagging = end_time - start_time

    print("Random search optimization runtime for BaggingRegressor:", runtime_random_search_bagging, "seconds")

    # Fit the RandomForestRegressor model with the best parameters
    start_time = time.time()
    random_forest_optimized = random_search_rf.best_estimator_
    random_forest_optimized.fit(X_train_raw, y_train)
    end_time = time.time()

    # Calculate the runtime for fitting the optimized RandomForestRegressor model
    runtime_random_forest = end_time - start_time

    print("RandomForestRegressor runtime:", runtime_random_forest, "seconds")

    # Fit the BaggingRegressor model with the best parameters
    start_time = time.time()
    bagging_optimized = random_search_bagging.best_estimator_
    bagging_optimized.fit(X_train_raw, y_train)
    end_time = time.time()

    # Calculate the runtime for fitting the optimized BaggingRegressor model
    runtime_bagging = end_time - start_time

    print("BaggingRegressor runtime:", runtime_bagging, "seconds")

    # Predict using the optimized RandomForestRegressor model
    optimized_pred_random_forest = random_forest_optimized.predict(X_test_raw)

    # Predict using the optimized BaggingRegressor model
    optimized_pred_bagging = bagging_optimized.predict(X_test_raw)

    # Evaluate the predictions using RMSE
    optimized_rmse_random_forest = mean_squared_error(y_test, optimized_pred_random_forest, squared=False)
    optimized_rmse_bagging = mean_squared_error(y_test, optimized_pred_bagging, squared=False)

    print("Optimized RMSE for RandomForestRegressor:", optimized_rmse_random_forest)
    print("Optimized RMSE for BaggingRegressor:", optimized_rmse_bagging)


random_reg()






