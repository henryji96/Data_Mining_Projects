# import warnings
# warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing




#import sys
#sys.path.append('../utils')

class PreProcessor(object):
    @staticmethod
    def drop_features(df, col_list):
        return df.drop(col_list, 1)

    @staticmethod
    def split_features_target(df, target_name):
        target = df[target_name]
        fetures = df.drop([target_name],1)
        return target, fetures

    @staticmethod
    def get_numeric_features(df):
#         numeric_type = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
#         features_numeric = df.select_dtypes(include=numeric_type)
        features_numeric = df._get_numeric_data()
        return features_numeric

    @staticmethod
    def get_categorical_features(df):

        features_categorical = df.select_dtypes(include='object')
        return features_categorical

    @staticmethod
    def train_test_split(X, y, test_size, random_state, stratify=True):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify = y)
        return X_train, X_test, y_train, y_test


class DiscreteFeaturesEncoder(object):

    @staticmethod
    def encode_ordinal(df,encode_map):
        """
        Assist to encode ordinal and binary class features.
        Input:
            {
                "feature_name1": {"good": 1, "bad": 0}
                "feature_name2": {"high": 3, "middel": 2, "low": 1}
            }
        """
        for feature_name, map_rule in encode_map.items():
            df[feature_name] = df[feature_name].map(map_rule)

        return df

    @staticmethod
    def one_hot_encoding(df, drop_first = True, columns = None, prefix_sep = '_', prefix = None ):
        """
        Encode nominal features
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html
        """
        return pd.get_dummies(df, columns = columns, drop_first = drop_first, prefix_sep = prefix_sep, prefix = prefix)


class FeatureTransformation(object):
    '''
    Transform / Scale / Normalize Numeric Features.
    '''
    @staticmethod
    def standard_scaler(df):
        '''
        Removing the mean value of each feature, then scale it by dividing non-constant features by their standard deviation.
        All features are centered around zero and have variance in the same order
        '''
        scaler = preprocessing.StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        return df_scaled

    @staticmethod
    def min_max_scaler(df):
        '''
        scaling features to lie between a given minimum and maximum value, often between zero and one
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        '''
        scaler = preprocessing.MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        return df_scaled

    @staticmethod
    def max_abs_scaler(df):
        '''
        Scales in a way that the training data lies within the range [-1, 1] by dividing through the largest maximum value in each feature.
        It is meant for data that is already centered at zero or sparse data.
        '''
        scaler = preprocessing.MaxAbsScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        return df_scaled

    @staticmethod
    def robust_scaler(df):
        '''
        This Scaler removes the median and scales the data according to the quantile range (defaults to IQR: Interquartile Range).
        '''
        scaler = preprocessing.RobustScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        return df_scaled

    @staticmethod
    def uniform_distribution_transformer(df):
        '''
        Non-linear Transformation
        '''
        transformer = preprocessing.QuantileTransformer(random_state=0)
        df_transformed = pd.DataFrame(transformer.fit_transform(df), columns=df.columns)
        return df_transformed

    @staticmethod
    def normal_distribution_transformer(df):
        '''
        Non-linear Transformation
        '''
        transformer = preprocessing.PowerTransformer(method='yeo-johnson')
        df_transformed = pd.DataFrame(transformer.fit_transform(df), columns=df.columns)
        return df_transformed

    @staticmethod
    def sample_normalizer(df):
        '''
        Normalize samples individually to unit norm.
        np.sqrt(np.sum(pow(x, 2))) == 1
        '''
        normalizer = preprocessing.Normalizer(norm='l2')
        df_normalized = pd.DataFrame(normalizer.fit_transform(df), columns = df.columns)
        return df_normalized




class Discretization(object):
    '''
    Continuous to Discrete
    '''
    @staticmethod
    def kbins_discretizer(one_feature, n_bins=3, strategy='uniform'):
        '''
        strategy
            uniform
            All bins in each feature have identical widths.
            quantile
            All bins in each feature have the same number of points.
            kmeans
            Values in each bin have the same nearest center of a 1D k-means cluster.
            Method used to encode the transformed result.
        '''
        kbd = preprocessing.KBinsDiscretizer(n_bins=[n_bins], encode='ordinal', strategy=strategy)
        feature_transformed = kbd.fit_transform(one_feature).reshape(-1,)
        print(kbd.n_bins_)
        print(kbd.bin_edges_)
        return feature_transformed

    def binarizer(one_feature, threshold):
        binarizer = preprocessing.Binarizer(threshold=3)
        feature_binarized = binarizer.fit_transform(one_feature).reshape(-1,)
        return feature_binarized



from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import statistics

class MissingValueImputer(object):

    @staticmethod
    def dropna(df):
        """
        axis : {0 or ‘index’, 1 or ‘columns’}, default 0
            0, or ‘index’ : Drop rows which contain missing values.
            1, or ‘columns’ : Drop columns which contain missing value.
        how : {‘any’, ‘all’}, default ‘any’
            ‘any’ : If any NA values are present, drop that row or column.
            ‘all’ : If all values are NA, drop that row or column.
        """
        return df.dropna(axis= 'index', how='any', inplace=False)

    @staticmethod
    def count_missing_value(df, missing_value=np.nan):
        """
        Count missing value for every features
        """
        print("Missing Value as '{}':".format(missing_value))
        cnt = df.applymap(lambda x: x==" ?").sum()
        cnt = cnt[cnt>0]
        for i in range(len(cnt)):
            print("| {:<10} {}".format(cnt[i], cnt.index[i]))

    @staticmethod
    def fillna_mode1(df, missing_value=np.nan):
        """
        Inplace
        missing_value: form of missing value
        """
        print("Missing value as '{}'".format(missing_value))
        cnt = df.applymap(lambda x: x==missing_value).sum()
        cnt_over0 = cnt[cnt>0]
        col_with_na = cnt_over0.index.values # get columns with missing value

        for i, col_name in enumerate(col_with_na):
            one_col = df[col_name]
            is_missing_value = np.array(one_col == missing_value)

            # fill missing value with (mode except for missing value)
            mode_without_na = statistics.mode(one_col[~is_missing_value])

            df.loc[is_missing_value, col_name] = mode_without_na

            print("  Feature: '{}'. Fill missing value with '{}'.".format(col_name, mode_without_na))

    @staticmethod
    def fillna_mode2(df, missing_values=np.nan):
        """
        Not Inplace
        SimpleImputer From Sklearn
        """
        imp = SimpleImputer(missing_values=missing_values,strategy="most_frequent")
        return pd.DataFrame(imp.fit_transform(df),columns = df.columns)

    @staticmethod
    def fillna_numeric(df, method='mean', value = None):
        """
        method : {‘backfill’, ‘bfill’, ‘pad’, ‘ffill’, None}, default None
            pad / ffill: propagate last valid observation forward to next valid
            backfill / bfill: use next valid observation to fill gap.

        own method: 'mean', 'median', 'zero'
        """
        if(method == 'mean'):
            value = df.mean()
            method = None
        elif(method=='median'):
            value = df.median()
            method = None
        elif(method=='zero'):
            value = 0
            method = None
        return df.fillna(value=value, method=method)

    @staticmethod
    def iterative_impute(df, estimator, missing_values = None):
        """
        estimator:
            BayesianRidge(),
            DecisionTreeRegressor(max_features='sqrt', random_state=0),
            ExtraTreesRegressor(n_estimators=10, random_state=0),
            KNeighborsRegressor(n_neighbors=15)
        """
        ite_imp = IterativeImputer(estimator = estimator, missing_values=missing_values,
                                    max_iter=10, imputation_order='ascending')
        df_imp = pd.DataFrame(ite_imp.fit_transform(df), columns = df.columns)
        return df_imp


from sklearn.feature_selection import REF
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
Class FeatureSelector(object):
    @staticmethod
    def variance_threshold(df, threshold):
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(df)
        selected_idx = selector.get_support(True)
        return df.iloc[:,selected_idx]

    @staticmethod
    def select_k_best(X, y, score_func, k):
        '''
        Supervised Feature Selection
        score_func classification:
            f_classif
            mutual_info_classif
            f_regression
        score_func regression:
            f_regression
            mutual_info_regression
        '''
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X, y)
        selected_idx = selector.get_support(True)

        selector_score = pd.DataFrame(data={
            "feature": X.columns,
            "score": selector.scores_,
            "pvalue":selector.pvalues_
        }).sort_values("score", ascending = False).reset_index(drop=True)
        display(selector_score.head(10))

        return df.iloc[:,selected_idx]

    @staticmethod
    def recursive_feature_elimination(X, y, estimator, n_features_to_select, step):
        '''
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
        step:
            If greater than or equal to 1, then step corresponds to the (integer) number of features to remove at each iteration.
            If within (0.0, 1.0), then step corresponds to the percentage (rounded down) of features to remove at each iteration.
        '''
        rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=step)
        rfe.fit(X, y)

        # print (rfe.ranking_)

        return X.loc[:,rfe.support_]



from sklearn.decomposition import PCA
class FeatureExtractior(object):
    @staticmethod
    def pca(X, n_components=None):
        '''
        X should be scaled
        '''
        pca = PCA(svd_solver='auto', n_components=n_components)
        X_pca = pca.fit_transform(X)

        # explained_variance_ratio = pca.explained_variance_ratio_
        # cum_explained_variance_ratio = np.cumsum(explained_variance_ratio)
        # singular_values_ = pca.singular_values_

        A = X.columns
        B = pca.components_[0]
        print ('first_pc = ')
        for i in range(len(B)):
            print("|  ", str(A[i])+'*'+str(B[i])+'+')
        return X_pca


from sklearn.utils import resample
class DataSampler(object):

    @staticmethod
    def random_sample(df, n_samples, replace=False):
        return resample(df, n_samples=n_sample, replace=replace, random_state=0)

    @staticmethod
    def stratified_sample(df, y, n_samples, replace=False):
        return resample(df, n_samples=n_sample, replace=replace, stratify=y, random_state=0)

    @staticmethod
    def stratified_sample2(X, y, n_sample, balanced=False, strict=False):
        '''
        If balance == False:
            sampled class is proportional to original class
        Else:
            if has enough data
                balanced
            elif no enough data
                all data for the class with small num of samples
            else
                strictly balanced
        '''
        if balanced == False:
            X_sample = resample(X, n_samples=n_sample,
                                replace=False, stratify=y, random_state=0)
            sample_idx = X_sample.index
            y_sample = y_train[sample_idx]
            return X_sample, y_sample

        if balanced == True:
            unique_class = y.unique()
            num_unique_class = len(unique_class)
            balanced_cnt_per_class = n_sample // num_unique_class

            enough_data = True
            lowest_num_class_cnt = y.value_counts().sort_values().reset_index(drop=True)[0]
            if lowest_num_class_cnt < balanced_cnt_per_class:
                enough_data = False

            sampled_idx = []
            for c in unique_class:
                index_one_class = X.loc[y==c,:].index.tolist()
                if enough_data == True:
                    sampled_idx_one_class = sample(index_one_class, balanced_cnt_per_class)
                elif strict == True:
                    sampled_idx_one_class = sample(index_one_class, lowest_num_class_cnt)
                else:
                    sampled_idx_one_class = sample(index_one_class, len(index_one_class))
                sampled_idx += sampled_idx_one_class
            return X.loc[sampled_idx,:], y.loc[sampled_idx]
