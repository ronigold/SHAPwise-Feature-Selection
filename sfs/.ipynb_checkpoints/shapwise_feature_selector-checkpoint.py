import numpy as np
import pandas as pd
import shap
import copy
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.preprocessing import MinMaxScaler

class SHAPwiseFeatureSelector(BaseEstimator):
    
    def __init__(self, base_estimator, metric, threshold=1.0, number_top_fi=20, strategy='max', verbose = True):
        self.base_estimator = base_estimator
        self.metric = metric
        self.threshold = threshold
        self.number_top_fi = number_top_fi
        self.strategy = strategy
        self.verbose = verbose
        self.features_to_drop = []

    def fit(self, X_train, y_train, X_valid, y_valid):
        self.base_estimator_ = self.clone_model()
        self.base_estimator_.fit(X_train, y_train)
        
        self.features_to_drop = self.identify_noisy_features(X_train, y_train, X_valid, y_valid)
        self.base_estimator_.fit(X_train.drop(columns=self.features_to_drop), y_train)
        if self.verbose:
            print('Summary:')
            print(f'Before drop: {str(self.original_score)}% After drop: {str(self.new_score)}%')
            improvement_percentage = (abs(self.new_score - self.original_score) / self.original_score) * 100
            print(f'Improvement of {round(improvement_percentage, 2)}%')

        return self
    
    def clone_model(self):
        return copy.deepcopy(self.base_estimator)

    def predict(self, X):
        return self.base_estimator_.predict(X.drop(columns=self.features_to_drop))

    def identify_noisy_features(self, X_train, y_train, X_valid, y_valid):
        """
        Identifies noisy features based on SHAP values and model performance.
        """
        noisy_features = []
        original_score = self.evaluate_performance(X_valid, y_valid, self.base_estimator_)
        self.original_score = original_score
        feature_impacts = self.calculate_shap_feature_impacts(self.base_estimator_, X_valid)['col_name']
        for feature in feature_impacts[:self.number_top_fi]:
            if self.is_noisy_feature(original_score, X_train, y_train, X_valid, y_valid, feature, noisy_features):
                if self.verbose:
                    print(f'Noisy feature detected: {feature} original_score (with {feature}): {original_score}, new score (without {feature}): {self.__temp_score}')
                    noisy_features.append(feature)
                    self.base_estimator_ = self.__temp_model
                    original_score = self.__temp_score
        self.new_score = original_score
        return noisy_features
    
    
    def is_noisy_feature(self, original_score, X_train, y_train, X_valid, y_valid, feature, excluded_features):
        """
        Determines if removing a feature improves the model's performance.
        """
        temp_features_to_exclude = excluded_features + [feature]
        self.__temp_model = self.fit_new_model(X_train.drop(columns=temp_features_to_exclude), y_train)
        self.__temp_score = self.evaluate_performance(X_valid.drop(columns=temp_features_to_exclude), y_valid, self.__temp_model)
        return self.is_new_score_better_by_strategy(original_score, self.__temp_score)
        
    def fit_new_model(self, X_train, y_train):
        return self.clone_model().fit(X_train, y_train)

    def evaluate_performance(self, X_valid, y_valid, model):
        """
        Evaluates the model's performance using the provided metric.
        """
        score = self.metric(y_valid, model.predict(X_valid))
        return score

    def is_new_score_better_by_strategy(self, original_score, new_score):
        if self.strategy == 'max':
            return new_score >= original_score
        else:
            return new_score <= original_score
    
    def get_shap_values(self, model, x):
        """
        Get SHAP values for a given model and dataset.
        - model: Trained model to analyze.
        - x: Input features DataFrame.
        """
        explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
        return explainer.shap_values(x)
    
    def calculate_shap_feature_impacts(self, model, x):
        """
        Calculate feature importance using SHAP values.
        - model: Trained model to analyze.
        - x: Input features DataFrame.
        """
        vals = self.get_shap_values(model, x)
        vals = np.abs(vals)
        if len(vals.shape) > 2: 
            vals = vals.mean(0)
        feature_importance = pd.DataFrame(list(zip(x.columns, sum(vals))), columns=['col_name', 'feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
        feature_importance['feature_importance_vals'] /= feature_importance['feature_importance_vals'].sum()
        return feature_importance.reset_index(drop=True)
