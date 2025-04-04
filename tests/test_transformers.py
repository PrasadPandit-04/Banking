import os, sys
import importlib
import pandas as pd
import numpy as np
from src.config import *
from src.preprocessing.transformers import *
from sklearn.utils.validation import check_is_fitted

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


######################################## Get pytest sample to test all transformers: ########################################

# csv_path = raw_data_dir/'Banking.csv'
csv_path = raw_data_dir/'Banking.csv'
df = pd.read_csv(csv_path, sep=';')
X = df.drop(['y'], axis=1)
y = df['y']

def get_samples(df):
    X = df.copy()
    index_of_unique_values_per_col = []
    unique_sample_index = []
    for col in X.select_dtypes(exclude='number').columns:
        index_of_unique_values_per_col.append([X[X[col] == value].index[0].tolist() for value in X[col].unique()])
    
    sample_index = list(set([items for sublist in index_of_unique_values_per_col for items in sublist]))
    # list(set(sample_index))
    pytest_sample = [X.loc[idx] for idx in set(sample_index)]
    return pd.DataFrame(pytest_sample, index=sample_index)

pytest_X = get_samples(X)
pytest_X.to_csv(f'{project_dir}/data/raw/pytest/pytest_X.csv', index=False)
pytest_y = y[pytest_X.index].apply(lambda x: 1 if x.lower() == 'yes' else 0 if x.lower() == 'no' else x)

######################################## Get pytest sample to test all transformers: ########################################

transformer = remove_placeholder()
pytest_X = pd.read_csv(f'{project_dir}/data/raw/pytest/pytest_X.csv')
result = transformer.fit_transform(pytest_X, pytest_y)
result.to_csv(f'{project_dir}/data/raw/pytest/remove_placeholder_expected.csv', index=False)

transformer = bin_transform()
pytest_X = pd.read_csv(f'{project_dir}/data/raw/pytest/remove_placeholder_expected.csv')
result = transformer.fit_transform(pytest_X, pytest_y)
result.to_csv(f'{project_dir}/data/raw/pytest/bin_transform_expected.csv', index=False)

transformer = ord_transform()
pytest_X = pd.read_csv(f'{project_dir}/data/raw/pytest/bin_transform_expected.csv')
result = transformer.fit_transform(pytest_X, pytest_y)
result.to_csv(f'{project_dir}/data/raw/pytest/ord_transform_expected.csv', index=False)

transformer = create_features()
pytest_X = pd.read_csv(f'{project_dir}/data/raw/pytest/ord_transform_expected.csv')
result = transformer.fit_transform(pytest_X, pytest_y)
result.to_csv(f'{project_dir}/data/raw/pytest/create_features_expected.csv', index=False)

class ColumnTransformerWithDataFrame(ColumnTransformer):
    def transform(self, X):
        X_transformed = super().transform(X)
        feature_names = super().get_feature_names_out()
        index = X.index
        return pd.DataFrame(X_transformed, columns=feature_names, index=index)

    def fit_transform(self, X, y=None):
        X_transformed = super().fit_transform(X)
        feature_names = super().get_feature_names_out()
        index = X.index
        return pd.DataFrame(X_transformed, columns=feature_names, index=index)

test_num_transformer = Pipeline(steps=[('scaler', num_transform()),])

test_cat_transformer = Pipeline(steps=[('encoder', cat_transform()),])

preprocessor = ColumnTransformerWithDataFrame(transformers=[
    ('num_features', test_num_transformer, num_features),
    ('cat_features', test_cat_transformer, cat_features),
], remainder='passthrough', force_int_remainder_cols=False, verbose_feature_names_out=True)

pipe = Pipeline(steps=[('preprocessor', preprocessor)])

pytest_X = pd.read_csv(f'{project_dir}/data/raw/pytest/create_features_expected.csv')
result = pipe.fit_transform(pytest_X, pytest_y)
# pd.testing.assert_frame_equal(result, expected)
result.to_csv(f'{project_dir}/data/raw/pytest/preprocessor_expected.csv', index=False)


transformer = impute_transform()
pytest_X = pd.read_csv(f'{project_dir}/data/raw/pytest/preprocessor_expected.csv')
result = transformer.fit_transform(pytest_X, pytest_y)
result.to_csv(f'{project_dir}/data/raw/pytest/impute_transform_expected.csv', index=False)

transformer = cal_drop_high_vif()
pytest_X = pd.read_csv(f'{project_dir}/data/raw/pytest/impute_transform_expected.csv')
result = transformer.fit_transform(pytest_X, pytest_y)
result.to_csv(f'{project_dir}/data/raw/pytest/cal_drop_high_vif_expected.csv', index=False)

transformer = feature_importance()
pytest_X = pd.read_csv(f'{project_dir}/data/raw/pytest/cal_drop_high_vif_expected.csv')
result = transformer.fit_transform(pytest_X, pytest_y)
result.to_csv(f'{project_dir}/data/raw/pytest/feature_importance_expected.csv', index=False)

transformer = feature_elimination()
pytest_X = pd.read_csv(f'{project_dir}/data/raw/pytest/feature_importance_expected.csv')
result = transformer.fit_transform(pytest_X, pytest_y)
result.to_csv(f'{project_dir}/data/raw/pytest/feature_elimination_expected.csv', index=False)

transformer = feature_selectkbest()
pytest_X = pd.read_csv(f'{project_dir}/data/raw/pytest/feature_elimination_expected.csv')
result = transformer.fit_transform(pytest_X, pytest_y)
result.to_csv(f'{project_dir}/data/raw/pytest/feature_selectkbest_expected.csv', index=False)


########################################  Define test column transformer with test pipe  ########################################

class ColumnTransformerWithDataFrame(ColumnTransformer):
    def transform(self, X):
        X_transformed = super().transform(X)
        feature_names = super().get_feature_names_out()
        index = X.index
        return pd.DataFrame(X_transformed, columns=feature_names, index=index)

    def fit_transform(self, X, y=None):
        X_transformed = super().fit_transform(X)
        feature_names = super().get_feature_names_out()
        index = X.index
        return pd.DataFrame(X_transformed, columns=feature_names, index=index)

test_num_transformer = Pipeline(steps=[('scaler', num_transform()),])

test_cat_transformer = Pipeline(steps=[('encoder', cat_transform()),])

preprocessor = ColumnTransformerWithDataFrame(transformers=[
    ('num_features', test_num_transformer, num_features),
    ('cat_features', test_cat_transformer, cat_features),
], remainder='passthrough', force_int_remainder_cols=False, verbose_feature_names_out=True)

pipe = Pipeline(steps=[('preprocessor', preprocessor)])

########################################  Test each transformer one by one  ########################################

def test_remove_placeholder():
    pytest_X = pd.read_csv(f'{project_dir}/data/raw/pytest/pytest_X.csv')
    expected = pd.read_csv(f'{project_dir}/data/raw/pytest/remove_placeholder_expected.csv')
    transformer = remove_placeholder()
    transformer.fit(pytest_X)  # Learn mode
    result = transformer.transform(pytest_X)
    pd.testing.assert_frame_equal(result, expected)      # Check if result matches expected

def test_bin_transform():
    pytest_X = pd.read_csv(f'{project_dir}/data/raw/pytest/remove_placeholder_expected.csv')
    expected = pd.read_csv(f'{project_dir}/data/raw/pytest/bin_transform_expected.csv')
    transformer = bin_transform()
    transformer.fit(pytest_X)  # Learn mode
    result = transformer.transform(pytest_X)
    pd.testing.assert_frame_equal(result, expected)      # Check if result matches expected

def test_ord_transform():
    pytest_X = pd.read_csv(f'{project_dir}/data/raw/pytest/bin_transform_expected.csv')
    expected = pd.read_csv(f'{project_dir}/data/raw/pytest/ord_transform_expected.csv')
    transformer = ord_transform()
    transformer.fit(pytest_X)
    result = transformer.transform(pytest_X)
    pd.testing.assert_frame_equal(result, expected)

def test_create_features():
    pytest_X = pd.read_csv(f'{project_dir}/data/raw/pytest/ord_transform_expected.csv')
    expected = pd.read_csv(f'{project_dir}/data/raw/pytest/create_features_expected.csv')
    transformer = create_features()
    transformer.fit(pytest_X)
    result = transformer.transform(pytest_X)
    pd.testing.assert_frame_equal(result, expected)

def test_preprocessor():
    pytest_X = pd.read_csv(f'{project_dir}/data/raw/pytest/create_features_expected.csv')
    expected = pd.read_csv(f'{project_dir}/data/raw/pytest/preprocessor_expected.csv')
    transformer = pipe
    transformer.fit(pytest_X)
    result = transformer.transform(pytest_X)
    # mean_ = np.isclose(result.mean(), 0)
    # std_ = np.isclose(result.std(), 1)
    # pd.testing.assert_extension_array_equal(mean_, std_)
    pd.testing.assert_frame_equal(result, expected)

# def test_cat_transform():
#     X = pd.read_csv(f'{project_dir}/data/raw/pytest/pytest_X.csv')
#     expected = pd.read_csv(f'{project_dir}/data/raw/pytest/remove_placeholder_expected.csv')
#     transformer = create_features()
#     transformer.fit(X)
#     result = transformer.transform(X)
#     pd.testing.assert_frame_equal(result, expected)

def test_impute_transform():
    pytest_X = pd.read_csv(f'{project_dir}/data/raw/pytest/preprocessor_expected.csv')
    expected = pd.read_csv(f'{project_dir}/data/raw/pytest/impute_transform_expected.csv')
    transformer = impute_transform()
    transformer.fit(pytest_X)
    result = transformer.transform(pytest_X)
    pd.testing.assert_frame_equal(result, expected)
    
def test_cal_drop_high_vif():
    pytest_X = pd.read_csv(f'{project_dir}/data/raw/pytest/impute_transform_expected.csv')
    expected = pd.read_csv(f'{project_dir}/data/raw/pytest/cal_drop_high_vif_expected.csv')
    transformer = cal_drop_high_vif(vif_threshold=10)  
    result = transformer.fit_transform(pytest_X)  
    # assert len(result.columns) < len(X.columns)
    pd.testing.assert_frame_equal(result, expected)

def test_feature_importance():
    pytest_X = pd.read_csv(f'{project_dir}/data/raw/pytest/cal_drop_high_vif_expected.csv')
    expected = pd.read_csv(f'{project_dir}/data/raw/pytest/feature_importance_expected.csv')
    transformer = feature_importance(threshold=0)  # Keep ≤ 80% features  
    result = transformer.fit_transform(pytest_X, pytest_y)  
    # assert len(result.columns) <= int(len(X.columns)*0.8)  # 80% of 5 = 4 (or fewer)
    print(result)
    pd.testing.assert_frame_equal(result, expected)

def test_feature_elimination():
    pytest_X = pd.read_csv(f'{project_dir}/data/raw/pytest/feature_importance_expected.csv')
    expected = pd.read_csv(f'{project_dir}/data/raw/pytest/feature_elimination_expected.csv')
    transformer = feature_elimination()  
    result = transformer.fit_transform(pytest_X, pytest_y)  
    pd.testing.assert_frame_equal(result, expected)
    # assert len(result.columns) <= int(len(X.columns)*0.8)   # threshold is defined in transformer 

def test_feature_selectkbest():
    pytest_X = pd.read_csv(f'{project_dir}/data/raw/pytest/feature_elimination_expected.csv')
    expected = pd.read_csv(f'{project_dir}/data/raw/pytest/feature_selectkbest_expected.csv')
    transformer = feature_selectkbest(k=25)  # given threshold 3 out of 5  
    result = transformer.fit_transform(pytest_X, pytest_y)
    pd.testing.assert_frame_equal(result, expected)
    # assert len(result.columns) <= 3


