from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import sys
#sys.path.append('../utils')

class Classifier(object):
    @staticmethod
    def train_all(X_train, X_test, y_train, y_test):
        lr_clf, coef_lr, score_lr = Classifier.logistic_reg(X_train, X_test, y_train, y_test)
        dt_clf, importance_dt, score_dt = Classifier.decision_tree_clf(X_train, X_test, y_train, y_test)
        rf_clf, importance_rf, score_rf = Classifier.random_forest_clf(X_train, X_test, y_train, y_test)
        gbdt_clf, importance_gbdt, score_gbdt = Classifier.gbdt_clf(X_train, X_test, y_train, y_test)
        score = pd.concat([score_lr, score_dt, score_rf, score_gbdt], 1)
        importance = pd.concat([coef_lr, importance_dt, importance_rf, importance_gbdt], 1)
        return score, importance

    @staticmethod
    def logistic_reg(X_train, X_test, y_train, y_test):
        clf = LogisticRegression()
        clf.fit(X_train, y_train)

        y_train_pred = clf.predict(X_train)
        y_train_pred_proba = clf.predict_proba(X_train)[:,1]
        y_test_pred = clf.predict(X_test)
        y_test_pred_proba = clf.predict_proba(X_test)[:,1]

        score_dict = {"LogisticReg_Train": {}, "LogisticReg_Test":  {}}
        score_dict['LogisticReg_Train']['f1'] = f1_score(y_train, y_train_pred)
        score_dict['LogisticReg_Train']['recall'] = recall_score(y_train, y_train_pred)
        score_dict['LogisticReg_Train']['precision'] = precision_score(y_train, y_train_pred)
        score_dict['LogisticReg_Train']['roc_auc_score'] = roc_auc_score(y_train, y_train_pred_proba)

        score_dict['LogisticReg_Test']['f1'] = f1_score(y_test, y_test_pred)
        score_dict['LogisticReg_Test']['recall'] = recall_score(y_test, y_test_pred)
        score_dict['LogisticReg_Test']['precision'] = precision_score(y_test, y_test_pred)
        score_dict['LogisticReg_Test']['roc_auc_score'] = roc_auc_score(y_test, y_test_pred_proba)
        score_df = pd.DataFrame(score_dict)[["LogisticReg_Train", "LogisticReg_Test"]]

        coef = pd.DataFrame(data = {
        'lr_feature': X_train.columns,
        'lr_coef_': clf.coef_[0]}).sort_values('lr_coef_',ascending=False).reset_index(drop=True)[['lr_feature','lr_coef_']]

        return clf, coef, score_df

    @staticmethod
    def decision_tree_clf(X_train, X_test, y_train, y_test):
        clf = DecisionTreeClassifier(random_state=0)
        clf.fit(X_train, y_train)

        y_train_pred = clf.predict(X_train)
        y_train_pred_proba = clf.predict_proba(X_train)[:,1]
        y_test_pred = clf.predict(X_test)
        y_test_pred_proba = clf.predict_proba(X_test)[:,1]

        score_dict = {"DT_Train": {}, "DT_Test":  {}}
        score_dict['DT_Train']['f1'] = f1_score(y_train, y_train_pred)
        score_dict['DT_Train']['recall'] = recall_score(y_train, y_train_pred)
        score_dict['DT_Train']['precision'] = precision_score(y_train, y_train_pred)
        score_dict['DT_Train']['roc_auc_score'] = roc_auc_score(y_train, y_train_pred_proba)

        score_dict['DT_Test']['f1'] = f1_score(y_test, y_test_pred)
        score_dict['DT_Test']['recall'] = recall_score(y_test, y_test_pred)
        score_dict['DT_Test']['precision'] = precision_score(y_test, y_test_pred)
        score_dict['DT_Test']['roc_auc_score'] = roc_auc_score(y_test, y_test_pred_proba)
        score_df = pd.DataFrame(score_dict)[["DT_Train", "DT_Test"]]

        importance = pd.DataFrame(data = {
        'dt_feature': X_train.columns,
        'dt_feature_importance': clf.feature_importances_.tolist()}).sort_values('dt_feature_importance',ascending=False).reset_index(drop=True)[['dt_feature','dt_feature_importance']]

        return clf, importance, score_df

    @staticmethod
    def random_forest_clf(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=3):
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth = max_depth, random_state=0)
        clf.fit(X_train, y_train)

        y_train_pred = clf.predict(X_train)
        y_train_pred_proba = clf.predict_proba(X_train)[:,1]
        y_test_pred = clf.predict(X_test)
        y_test_pred_proba = clf.predict_proba(X_test)[:,1]

        score_dict = {"RF_Train": {}, "RF_Test":  {}}
        score_dict['RF_Train']['f1'] = f1_score(y_train, y_train_pred)
        score_dict['RF_Train']['recall'] = recall_score(y_train, y_train_pred)
        score_dict['RF_Train']['precision'] = precision_score(y_train, y_train_pred)
        score_dict['RF_Train']['roc_auc_score'] = roc_auc_score(y_train, y_train_pred_proba)

        score_dict['RF_Test']['f1'] = f1_score(y_test, y_test_pred)
        score_dict['RF_Test']['recall'] = recall_score(y_test, y_test_pred)
        score_dict['RF_Test']['precision'] = precision_score(y_test, y_test_pred)
        score_dict['RF_Test']['roc_auc_score'] = roc_auc_score(y_test, y_test_pred_proba)
        score_df = pd.DataFrame(score_dict)[["RF_Train", "RF_Test"]]

        importance = pd.DataFrame(data = {
        'rf_feature': X_train.columns,
        'rf_feature_importance': clf.feature_importances_.tolist()}).sort_values('rf_feature_importance',ascending=False).reset_index(drop=True)[['rf_feature','rf_feature_importance']]

        return clf, importance, score_df

    @staticmethod
    def gbdt_clf(X_train, X_test, y_train, y_test):
        clf = GradientBoostingClassifier(n_estimators=2000,learning_rate=0.01)
        clf.fit(X_train, y_train)

        y_train_pred = clf.predict(X_train)
        y_train_pred_proba = clf.predict_proba(X_train)[:,1]
        y_test_pred = clf.predict(X_test)
        y_test_pred_proba = clf.predict_proba(X_test)[:,1]

        score_dict = {"GBDT_Train": {}, "GBDT_Test":  {}}
        score_dict['GBDT_Train']['f1'] = f1_score(y_train, y_train_pred)
        score_dict['GBDT_Train']['recall'] = recall_score(y_train, y_train_pred)
        score_dict['GBDT_Train']['precision'] = precision_score(y_train, y_train_pred)
        score_dict['GBDT_Train']['roc_auc_score'] = roc_auc_score(y_train, y_train_pred_proba)

        score_dict['GBDT_Test']['f1'] = f1_score(y_test, y_test_pred)
        score_dict['GBDT_Test']['recall'] = recall_score(y_test, y_test_pred)
        score_dict['GBDT_Test']['precision'] = precision_score(y_test, y_test_pred)
        score_dict['GBDT_Test']['roc_auc_score'] = roc_auc_score(y_test, y_test_pred_proba)
        score_df = pd.DataFrame(score_dict)[["GBDT_Train", "GBDT_Test"]]

        importance = pd.DataFrame(data = {
        'gbdt_feature': X_train.columns,
        'gbdt_feature_importance': clf.feature_importances_.tolist()}).sort_values('gbdt_feature_importance',ascending=False).reset_index(drop=True)[['gbdt_feature','gbdt_feature_importance']]

        return clf, importance, score_df


    def xgb_clf(X_train, X_test, y_train, y_test, booster = 'gbtree', learning_rate=0.1,
                n_estimators=100, max_depth=3, \
                reg_alpha=0, reg_lambda=0, n_jobs = -1, subsample = 1, colsample_bytree = 1):
        '''
        https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
        '''
        clf = XGBClassifier(n_estimators=n_estimators,
                            n_jobs = n_jobs,
                            max_depth = max_depth,
                            reg_alpha = reg_alpha,
                            reg_lambda = reg_lambda,
                            learning_rate = learning_rate,
                            subsample = subsample,
                            colsample_bytree = colsample_bytree,
                            random_state=50)
        clf.fit(X_train, y_train)

        y_train_pred = clf.predict(X_train)
        y_train_pred_proba = clf.predict_proba(X_train)[:,1]
        y_test_pred = clf.predict(X_test)
        y_test_pred_proba = clf.predict_proba(X_test)[:,1]

        score_dict = {"XGB_Train": {}, "XGB_Test":  {}}
        score_dict['XGB_Train']['f1'] = f1_score(y_train, y_train_pred)
        score_dict['XGB_Train']['recall'] = recall_score(y_train, y_train_pred)
        score_dict['XGB_Train']['precision'] = precision_score(y_train, y_train_pred)
        score_dict['XGB_Train']['roc_auc_score'] = roc_auc_score(y_train, y_train_pred_proba)

        score_dict['XGB_Test']['f1'] = f1_score(y_test, y_test_pred)
        score_dict['XGB_Test']['recall'] = recall_score(y_test, y_test_pred)
        score_dict['XGB_Test']['precision'] = precision_score(y_test, y_test_pred)
        score_dict['XGB_Test']['roc_auc_score'] = roc_auc_score(y_test, y_test_pred_proba)
        score_df = pd.DataFrame(score_dict)[["XGB_Train", "XGB_Test"]]

        importance = pd.DataFrame(data = {
        'xgb_feature': X_train.columns,
        'xgb_feature_importance': clf.feature_importances_.tolist()}).sort_values('xgb_feature_importance',ascending=False).reset_index(drop=True)[['xgb_feature','xgb_feature_importance']]

        return clf, importance, score_df


    @staticmethod
    def classification_metrics(y_true, y_pred, y_pred_proba):

        roc_auc = roc_auc_score(y_true, y_pred_proba)
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print("roc_auc: ")
        print(roc_auc)
        print()

        print("classification_report: \n")
        report = classification_report(y_true, y_pred)
        print(report)
        print()


        labels = np.unique(y_true)
        index = ['true: {}'.format(idx) for idx in labels]
        column = ['pred: {}'.format(idx) for idx in labels]

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, columns = column, index = index)

        cm_normalized_df =  cm_df.div(cm_df.sum(axis=1), axis=0)
        print("confusion_matrix: \n")
        print(cm_df)
        print("\n")
        print("normalized_confusion_matrix: \n")
        print(cm_normalized_df)

    @staticmethod
    def predict_proba_analysis(y_true, y_pred_proba, threshold = [0.3, 0.5, 0.8], plot_proba_violin = True, plot_thresohld_score=True):
            result = {'f1':[], 'recall':[], 'precision':[]}

            roc_auc = roc_auc_score(y_true, y_pred_proba)
            print(roc_auc)

            # different cutting point
            for th in threshold:
                y_pred_label = list(map(lambda x: 0 if x < th else 1, y_pred_proba))
                result['f1'].append(f1_score(y_true, y_pred_label))
                result['precision'].append(precision_score(y_true, y_pred_label))
                result['recall'].append(recall_score(y_true, y_pred_label))
            result_df = pd.DataFrame(result, index = threshold)

            if plot_thresohld_score == True:
                result_df.plot()
                plt.xticks(np.arange(0.4, 0.6, 0.02))
                plt.show()

            if plot_proba_violin == True:
                sns.violinplot(y = y_pred_proba, x = y_true)
                plt.show()
            return result_df

    @staticmethod
    def pred_with_threshold(pred_proba, threshold):
        return list(map(lambda x: 0 if x < threshold else 1, pred_proba))
