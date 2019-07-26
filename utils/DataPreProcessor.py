import warnings
warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import numpy as np
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
    def encode_category(df,encode_map):
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
    # https://blog.csdn.net/brucewong0516/article/details/80406564
    def fillna_numeric(df, method='mean'):
        if(all_numeric):
            if(method == 'mean'):
                return df.fillna(df.mean())
            elif(method=='zero'):
                return df.fillna(0)
        else:
            df_numeric = get_numeric_features(df)
            df_category = get_categorical_features(df)
   
    @staticmethod
    def fillna_numeric(df, method='mean', reorder = False):
        """
        (reorder = True) means put categorical columns to the left; numeric columns to the right
        """
        if reorder == False:
            if(method == 'mean'):
                return df.fillna(df.mean())
            elif(method=='zero'):
                return df.fillna(0)
        elif reorder == True:
            df_numeric = PreProcessor.get_numeric_features(df)
            df_other = df.drop(df_numeric.columns, 1)
            if(method == 'mean'):
                df_numeric_clean = df_numeric.fillna(df.mean())
            elif(method=='zero'):
                df_numeric_clean = df_numeric.fillna(0)

            return pd.concat([df_other, df_numeric_clean], 1)

    
    
    @staticmethod    
    def fillna_mode(df, missing_values=np.nan):
        """
        https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer
        """
        imp = SimpleImputer(missing_values=missing_values,strategy="most_frequent")          
        return pd.DataFrame(imp.fit_transform(df),columns = df.columns)
    
    @staticmethod
    def one_hot_encoding(df, columns = None, prefix_sep = '_', prefix = None ):
        """
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html
        """
        return pd.get_dummies(df, columns = columns, prefix_sep = prefix_sep, prefix = prefix)
    
    @staticmethod
    def train_test_split(X, y, test_size, random_state):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test