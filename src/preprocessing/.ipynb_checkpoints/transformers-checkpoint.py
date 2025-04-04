
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from statsmodels.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif

from src.config import *
###############################################################################################################################################################

class ColumnTransformerWithDataFrame(ColumnTransformer):
    def fit(self, X, y=None):
        super().fit(X)
        return self
        
    def transform(self, X):
        X_transformed = super().transform(X)
        # feature_names = super().get_feature_names_out()
        feature_names = [super().get_feature_names_out()[i].split('__')[1] for i in range(len(super().get_feature_names_out()))]
        # print('ColumnTransformerWithDataFrame --> fit', feature_names)
        index = X.index
        return pd.DataFrame(X_transformed, columns=feature_names, index=index)

    def fit_transform(self, X, y=None):
        X_transformed = super().fit_transform(X)
        # feature_names = super().get_feature_names_out()
        feature_names = [super().get_feature_names_out()[i].split('__')[1] for i in range(len(super().get_feature_names_out()))]
        # print('ColumnTransformerWithDataFrame --> transform', feature_names)
        index = X.index
        return pd.DataFrame(X_transformed, columns=feature_names, index=index)

###############################################################################################################################################################

class remove_placeholder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, 'fitted_')
        X = X.copy()
        # X.replace(to_replace='unknown', value = pd.to_numeric(np.nan, errors='coerce'), inplace=True)
        X.replace(to_replace='unknown', value = np.nan, inplace=True)
        # fill_value = {'unknown' : np.nan}
        # X_temp.where(X_temp.apply(lambda x: x!='unknown'), np.nan, inplace=True)
        # X_temp.fillna(fill_value)

        # X_temp_mode = {}
        # for col in X_temp.columns:
        #     X_temp_mode[col] = X_temp[col].mode()[1] if pd.isna(X_temp[col].mode()[0]) else X_temp[col].mode()[0]
        #     # print(col, X_temp_mode[col])
        mode = {key: item[0] for key, item in X.mode().to_dict().items()}
        X.fillna(mode, inplace=True)
        # print(X_temp_mode)
        
        # print(f'Placeholder unknown replaced by np.nan.')
        print('\n---------------------------------------------------------------------------------------------')
        # print('\n\n remove_placeholder - head of X: \n',X.head(5))
        print(f'\n\tRemove unknown Placeholder:\t Total Features: {X.shape[1]}\t Total Samples: {X.shape[0]}')
        # print('\n---------------------------------------------------------------------------------------')
        return X

###############################################################################################################################################################

class bin_transform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.cols = bin_features
        self.fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, 'fitted_')
        X = X.copy()
        for col in self.cols:
            # X[col] = X[col].map({'yes': 1, 'no': 0, })
            # X[col] = X[col].apply(lambda x: pd.to_numeric(np.nan, errors='coerce') if type(x) == float else 1 if x.lower() == 'yes' else 0 if x.lower() == 'no' else x)
            X[col] = X[col].apply(lambda x: 1 if x.lower() == 'yes' else 0 if x.lower() == 'no' else x)
        
        # print(f'Binary features transformed: {X.columns.to_list()}')
        # print('\n---------------------------------------------------------------------------------------')
        # print('\n\n bin_transform - head of X: \n',X.head(5))
        print(f'\tBinary Transform:\t\t Total Features: {X.shape[1]}\t Total Samples: {X.shape[0]}')
        # print('\n---------------------------------------------------------------------------------------')
        return X

###############################################################################################################################################################

class ord_transform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, 'fitted_')
        X = X.copy()
        X['education'] = X['education'].map({'unknown': pd.to_numeric(np.nan, errors='coerce'), 'primary': 1, 'secondary': 2, 'tertiary': 3}).fillna(-1)
        X['month'] = X['month'].map({'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}).fillna(-1)
        # X['job'] = X['job'].map({'management': 0, 'entrepreneur': 1, 'self-employed': 2, 'technician': 3, 'blue-collar': 4, 'admin.': 5, 'services': 6, 'retired': 7, 'unemployed': 8, 'housemaid': 9, 'unknown': 10})
        # print(f'Ordinal features transformed: {['education', 'month']}')
        # print('\n---------------------------------------------------------------------------------------')
        # print('\n\n ord_transform - head of X: \n',X.head(5))
        print(f'\tOrdinary Feature Transform:\t Total Features: {X.shape[1]}\t Total Samples: {X.shape[0]}')
        # print('\n---------------------------------------------------------------------------------------')
        return X

###############################################################################################################################################################

class create_features(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.num_cols = None
        self.cat_cols = None

    def fit(self, X, y=None):
        self.num_cols = X.select_dtypes(np.number).columns
        self.cat_cols = X.select_dtypes(object).columns
        # print(f'num_cols in create_features :{self.num_cols}')
        # print(f'cat_cols in create_features :{self.cat_cols}')
        self.fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, 'fitted_')
        self.num_cols = X.select_dtypes(np.number).columns
        self.cat_cols = X.select_dtypes(object).columns
        X_new = X.copy()
        #######################################################
        # print('Total cat features: ', len(self.cat_cols))
        # print('Total num features: ', len(self.num_cols))
        # total_features = []
        # new_cols = {}
        # for i, i_col in enumerate(self.num_cols):
        #     for j, j_col in enumerate(self.num_cols):
        #         if j > i:
        #             new_cols[f'{i_col}_{j_col}'] = X[i_col] * X[j_col]
        #             # print(f'{i_col}_{j_col} feature added to X')
        #             total_features.append(f'{i_col}_{j_col}')
 
        # for i, i_col in enumerate(self.cat_cols):
        #     for j, j_col in enumerate(self.cat_cols):
        #         if j > i:
        #             new_cols[f'{i_col}_{j_col}'] = X[i_col] + X[j_col]
        #             # print(f'{i_col}_{j_col} feature added to X')
        #             total_features.append(f'{i_col}_{j_col}')
        #######################################################

        today = dt.datetime.today()
        X_new['days_since_last_contact'] = X[['day', 'month']].apply(
            lambda x: (today - dt.datetime(year=today.year - 1, month=int(x['month']), day=int(x['day']))).days,axis=1)
        
        # print('No. of total features added: ', len(total_features)+1)
        # # print('Total features added: ', total_features, 'days_since_last_contact')

        # new_cols_df = pd.DataFrame(new_cols, columns=new_cols.keys(), index=X_new.index)
        # print('Does index of new_cols_df and X_new matches:', X_new.index.equals(new_cols_df.index))

        ########################################### One-Hot Encode for new interaction features ###################################
        # ohe = OneHotEncoder(drop='first', handle_unknown='ignore', max_categories=50, min_frequency=0.1)
        # ohe.fit(new_cols_df)
        # ohe_data = ohe.transform(new_cols_df)
        # new_cols_df = (pd.DataFrame(ohe_data.toarray(), columns=ohe.get_feature_names_out()))
        # new_cols_df = new_cols_df.dropna(axis=1, how='all')
        # print('X_new.shape: ', X_new.shape)  # Expect (7088, n_features)
        # print('new_cols_df.shape: ', new_cols_df.shape)  # Expect (7088, m_features)
        # X_new = X_new.join(new_cols_df)
        ###########################################       F1-score: 0.4286       ###########################################

        ########################################### Frequency Encode for new interaction features ###################################
        # new_cat_cols = new_cols_df.select_dtypes(object).columns
        # for col in new_cat_cols:
        #     freq_encoding = new_cols_df[col].value_counts() / len(new_cols_df)
        #     new_cols_df[col] = new_cols_df[col].map(freq_encoding)
        # X_new = X_new.join(new_cols_df)
        ###########################################       F1-score: 0.2031       ###########################################

        ########################################### Label Encode for new interaction features ###################################
        # new_cat_cols = new_cols_df.select_dtypes(object).columns
        # le = LabelEncoder()
        # for col in new_cat_cols:
        #     new_cols_df[col] = le.fit_transform(new_cols_df[col])
        # X_new = X_new.join(new_cols_df)
        ###########################################       F1-score: 0.4449       ###########################################
        
        # print('\n---------------------------------------------------------------------------------------')
        # print('\n\n create_features - head of X_new: \n',X_new.head(5))
        print(f'\tFeature Engineering:\t\t Total Features: {X_new.shape[1]}\t Total Samples: {X_new.shape[0]}')
        # print('\n---------------------------------------------------------------------------------------')
        return X_new

###############################################################################################################################################################

class num_transform(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        self.cols_to_scale = None
        self.scaled_cols = None
        self.final_df = None

    def fit(self, X, y=None):
        self.total_cols = X.columns.to_list()
        # print('Total Cols: ',self.total_cols)
        self.scaler.fit(X[self.total_cols])
        self.scaled_cols = self.scaler.get_feature_names_out()
        self.fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, 'fitted_')
        X = X.copy()
        index = X[self.scaled_cols].index
        # print(f'Numerical features before scaling: \n{X.head(2)}')
        X_scaled = self.scaler.transform(X[self.scaled_cols])
        X_scaled = pd.DataFrame(X_scaled, columns=self.scaled_cols, index=index)
        X_scaled_df = pd.concat([X.drop(self.scaled_cols, axis=1), X_scaled], axis=1)
        # print(f'Numerical features after scaling: \n{X_scaled_df.columns.to_list()}')
        # print('\n---------------------------------------------------------------------------------------')
        # print('\n\n num_transform - head of X_scaled_df: \n',X_scaled_df.head(5))
        print(f'\tNumerical Feature scaling:\t Total Features: {X_scaled_df.shape[1]}\t Total Samples: {X_scaled_df.shape[0]}')
        # print('\n---------------------------------------------------------------------------------------')
        self.final_df = X_scaled_df
        return X_scaled_df

    def get_feature_names_out(self, input_features=None):
        # Return the feature names after transformation
        if hasattr(self.final_df, 'columns'):
            return self.final_df.columns
        else:
            # return [f"feature_{i}" for i in range(self.final_df.shape[1])]
            temp = [x for x in self.total_cols if x not in self.cols_to_scale]
            temp.extend(self.scaled_cols)
            return temp

###############################################################################################################################################################

class cat_transform(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        self.cols_to_en = None
        self.encoded_cols = None
        self.final_df = None

    def fit(self, X, y=None):
        # print(f'in cat_transform fit (X cols are): {X.columns}')
        self.total_cols = X.columns.to_list()
        self.cols_to_en = X.select_dtypes(object).columns
        self.encoder.fit(X[self.cols_to_en])
        self.encoded_cols = self.encoder.get_feature_names_out(self.cols_to_en)
        # print("Categories found during fit: ", self.encoder.categories_)
        self.fitted_ = True
        return self
        
    def transform(self, X):
        check_is_fitted(self, 'fitted_')
        # print(f'in cat_transform transform (X cols are): {X.columns}')
        X = X.copy()
        X_en = self.encoder.transform(X[self.cols_to_en])
        X_en = pd.DataFrame(X_en, columns=self.encoded_cols, index=X.index)
        X_en_df = pd.concat([X.drop(self.cols_to_en, axis=1), X_en], axis=1)
        
        # print('\n---------------------------------------------------------------------------------------')
        # print(f'\n After cat_transform shape:{X_en_df.shape}')
        print(f'\tCategorical Feature Transform:\t Total Features: {X_en_df.shape[1]}\t Total Samples: {X_en_df.shape[0]}')
        # print('\n---------------------------------------------------------------------------------------')
        # print('\n\n cat_transform - head of X_en_df: \n',X_en_df.head(5))
        self.final_df = X_en_df
        return X_en_df

    def get_feature_names_out(self, input_features=None):
        # Return the feature names after transformation
        if hasattr(self.final_df, 'columns'):
            return self.final_df.columns
        else:
            # return [f"feature_{i}" for i in range(self.final_df.shape[1])]
            temp = [x for x in self.total_cols if x not in self.cols_to_en]
            temp.extend(self.encoded_cols)
            return temp

###############################################################################################################################################################

class impute_transform(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imputer = IterativeImputer(random_state=42)
        self.cols_to_impute = None
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        self.fitted_ = True
        # X = pd.DataFrame(X, columns=preprocessor.get_feature_names_out())
        # print('\n In impute_transform fit\n', X.shape)
        self.imputer.fit(X)
        # self.cols_to_impute = preprocessor.get_feature_names_out()
        self.feature_names_out_ = X.columns
        return self

    def transform(self, X):
        check_is_fitted(self, 'fitted_')
        X = X.copy()
        # X = pd.DataFrame(X, columns=self.cols_to_impute)
        # print(f'\n Before Imputer X shape:{X.shape}')
        X_imputed = self.imputer.transform(X)

        X_imputed_df = pd.DataFrame(X_imputed, columns=self.feature_names_out_)        
        # print(f'Imputation done for features: \n{self.imputer.get_feature_names_out()}')
        # print('\n---------------------------------------------------------------------------------------')
        # print('\n\n impute_transform - head of X_imputed_df: \n',X_imputed_df.head(5))
        # print(f'\n After Imputer shape:{X_imputed_df.shape}')
        # print(f'\n Null values:{X_imputed_df.isnull().sum()}')
        print(f'\tImputed Dataset:\t\t Total Features: {X_imputed_df.shape[1]}\t Total Samples: {X_imputed_df.shape[0]}')
        # print('\n---------------------------------------------------------------------------------------')
        return X_imputed_df

###############################################################################################################################################################

class cal_drop_high_vif(BaseEstimator, TransformerMixin):
    def __init__(self, vif_threshold=10):
        self.vif_df = pd.DataFrame()
        self.vif_threshold = vif_threshold
        self.cols_to_drop = []

    def fit(self, X, y=None):
        temp_X = X.copy()
        vif_df = self.cal_vif(temp_X)
        while vif_df['VIF'].max() > self.vif_threshold:
            drop_col = vif_df.sort_values(by='VIF', ascending=False).iloc[0]['Feature']
            temp_X = temp_X.drop(drop_col, axis=1)
            self.cols_to_drop.append(drop_col)
            vif_df = self.cal_vif(temp_X)
        # print('Cols_to_drop : ', self.cols_to_drop)
        self.fitted_ = True
        return self
    
    def cal_vif(self, X):
        X_const = add_constant(X)
        vif_df = pd.DataFrame()
        vif_df['Feature'] = X_const.columns
        vif_df['VIF'] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
        # print(self.vif_df)
        return vif_df.drop(vif_df.index[vif_df['Feature'] == 'const'], axis=0)

    def transform(self, X):
        check_is_fitted(self, 'fitted_')
        X.drop(self.cols_to_drop, axis=1, inplace=True)
        # vif_df = pd.DataFrame()
        # vif_df = self.cal_vif(X)
        # removed_features = []
        # while vif_df['VIF'].max() > self.vif_threshold:
        #     feature_to_remove = vif_df.sort_values(by='VIF', ascending=False).iloc[0]['Feature']
        #     removed_features.append(feature_to_remove)
        #     # print(f'Feature removed: {feature_to_remove}')
        #     X = X.drop(columns=[feature_to_remove], axis=1)
        #     vif_df = self.cal_vif(X)
        # print('\n---------------------------------------------------------------------------------------')
        # print(f'\n Removed features:\t {self.cols_to_drop}')
        # print(f'\n VIF Data:\n {vif_df}')
        # X = pd.DataFrame(X, self.feature_names_out_)
        print(f'\tAfter VIF:\t\t\t Total Features: {X.shape[1]}\t Total Samples: {X.shape[0]}')
        # print('\n---------------------------------------------------------------------------------------------\n')
        return X

###############################################################################################################################################################

class feature_importance(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0):
        self.fi = XGBClassifier(scale_pos_weight=7, importance_type='weight')
        self.sel_cols = []
        self.threshold = threshold

    def fit(self, X, y=None):
        self.fi.fit(X, y)
        # self.sel_cols = [X.columns.to_list()[i] for i in range(len(X.columns.to_list())) if self.fi.feature_importances_[i]>self.threshold]
        self.sel_cols = [feature for feature, importance in zip(X.columns, self.fi.feature_importances_) if importance >= self.threshold]
        # print('FI Fit: ', self.sel_cols)
        # print('FI: ', self.fi.feature_importances_)
        self.fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, 'fitted_')
        # self.fi.fit(X, y)
        X = X[self.sel_cols].copy()
        # print('\n---------------------------------------------------------------------------------------------\n')
        # X = pd.DataFrame(X[self.sel_cols], columns=self.sel_cols)
        print(f'\tAfter Feature Importance:\t Total Features: {X.shape[1]}\t Total Samples: {X.shape[0]}')
        # print('\n---------------------------------------------------------------------------------------------\n')
        return X

###############################################################################################################################################################

class feature_elimination(BaseEstimator, TransformerMixin):
    def __init__(self, n_features_to_select = 1.0):
        self.n_features_to_select = n_features_to_select
        self.rfe = RFE(estimator=XGBClassifier(importance_type='weight'), n_features_to_select=self.n_features_to_select, step=1)
        self.sel_cols = None

    def fit(self, X, y=None):
        self.rfe.fit(X, y)
        self.sel_cols = self.rfe.get_feature_names_out()
        # print('RFE Fit:')
        self.fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, 'fitted_')
        X_rfe = self.rfe.transform(X)
        X_rfe_df = pd.DataFrame(X_rfe, columns=self.sel_cols)
        # print('\n---------------------------------------------------------------------------------------------\n')
        print(f'\tAfter RFE:\t\t\t Total Features: {X_rfe_df.shape[1]}\t Total Samples: {X_rfe_df.shape[0]}')
        # print('\n---------------------------------------------------------------------------------------------\n')
        return X_rfe_df

###############################################################################################################################################################

class feature_selectkbest(BaseEstimator, TransformerMixin):
    def __init__(self, k='all'):
        self.k = k
        self.kbest = SelectKBest(score_func=f_classif, k=self.k)
        self.sel_cols = None

    def fit(self, X, y=None):
        self.kbest.fit(X, y)
        self.sel_cols = self.kbest.get_feature_names_out()
        # print('\nfeature_selectkbest fit', self.sel_cols)
        self.fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, 'fitted_')
        X_kbest = self.kbest.transform(X)
        X_kbest_df = pd.DataFrame(X_kbest, columns=self.sel_cols)
        # print('\nfeature_selectkbest transform')
        # print('\n---------------------------------------------------------------------------------------------\n')
        print(f'\tAfter SelectKBest:\t\t Total Features: {X_kbest_df.shape[1]}\t Total Samples: {X_kbest_df.shape[0]}')
        print('\n---------------------------------------------------------------------------------------------\n')
        return X_kbest_df

###############################################################################################################################################################
