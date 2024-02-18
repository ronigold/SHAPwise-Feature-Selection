import shap
import copy
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple, Union

class SHAPwiseFeatureSelector(BaseEstimator):
    """
    A feature selector that uses SHAP values to identify and retain the most impactful features.
    
    Parameters:
    - model: The machine learning model for which SHAP values are calculated.
    - threshold: The minimum mean absolute SHAP value a feature must have to be retained.
    """
    def __init__(self, base_estimator, metric, threshold=1.0, number_top_fi=20, min_or_max = 'max'):
        self.base_estimator = base_estimator
        self.threshold = threshold
        self.number_top_fi = number_top_fi
        self.min_or_max = min_or_max
        self.metric = metric
        self.features_to_drop = []

    def fit(self, X_train, y_train, X_valid, y_valid):
        """
        Fit the feature selector to the data.
        
        Parameters:
        - X: Feature data as a pandas DataFrame.
        - y: Target variable. Not used in this method but included for compatibility.
        """
        # Determine which features exceed the threshold
        self.drop_features = self.__get_noisy_features(
                        self.base_estimator,
                        X_train,
                        y_train,
                        X_valid,
                        y_valid,
                        self.metric,
                        self.min_or_max,
                        self.number_top_fi,
                        self.threshold
                    )
        self.base_estimator.fit(X_train.drop(self.drop_features, axis = 1), y_train)
        return self

    def predict(self, X):
        """
        Transform the dataset to include only the selected features.
        
        Parameters:
        - X: Feature data as a pandas DataFrame.
        
        Returns:
        - A DataFrame containing only the selected features.
        """
        # Select features that meet the threshold
        return X.iloc[:, self.important_features]

    def get_base_estimator(self):
        """
        Fit to data, then transform it.
        
        Parameters:
        - X: Feature data as a pandas DataFrame.
        - y: Target variable. Not used in this method but included for compatibility.
        
        Returns:
        - A DataFrame containing only the selected features.
        """
        return self.base_estimator
    
    def get_features_to_drop(self):
        return self.features_to_drop
    
    def __get_noisy_features(
        self,
        model,
        x_train,
        y_train,
        x_valid,
        y_valid,
        metric,
        min_or_max,
        number_top_fi,
        distance
    ) -> List[str]:
        """Feature reduction method based on SHAP.

        Full multi-dimensional support. The starting point is that the higher the correlation
        between SHAP values and the original values, the more significant this feature is, and the
        lower this correlation, the more likely it is to over-match.

        The function detects suspicious features by calculating the distances between the Shap
        values and the original values and tries to remove them, if after removal the score is
        higher, this function is added to the removal list and the procedure is repeated.

        Args:
            X: Input Dataset.
            y: Input target.
            distance: Represents between_shap_populations is a value between [0,1], and roughly
                translates to how "separate" we demand our SHAP populations to be.
            number_top_fi: The amount of features on which the calculation is performed in each
                iteration.
            opt_metric: Optimization metric. If None, will use the default set in the class
                instance.

        Returns:
            List of noisy features the algorithm found.
        """
        def get_noisy_features_iteration(
            model: BaseEstimator,
            x: pd.DataFrame,
            number_top_fi: int = 20,
            distance: Union[float, int] = 1,
        ) -> pd.DataFrame:
            """Feature reduction method based on SHAP.

            Full multi-dimensional support.

            TODO: Add dimension name.

            Args:
                model: The model to get the noisy features from.
                x: The feature data.
                number_top_fi: The amount of features on which the calculation is performed in
                    each iteration.
                distance: Represents between_shap_populations is a value between [0,1], and
                    roughly translates to how "separate" we demand our SHAP populations to be.

            Returns:
                A data frame with the noisy features.
            """

            def get_noisy_features_per_dim(
                vals: shap.Explanation, x: pd.DataFrame, fi_list: List[str], number_top_fi: int
            ) -> List[np.ndarray]:
                def normalization_df(df: pd.DataFrame) -> pd.DataFrame:
                    min_max_scaler = preprocessing.MinMaxScaler()
                    x_scaled = min_max_scaler.fit_transform(df)
                    return pd.DataFrame(x_scaled, columns=df.columns)

                df_shap = pd.DataFrame(vals, columns=x.columns.values).drop(
                    fi_list[number_top_fi:], axis=1
                )

                # TEMPORARY SOLUTION - only work with numeric cols.
                # relevant for cases where we didn't one-hot encode the features, e.g. for lightGBM
                numeric_cols = [c for c in x.columns if str(x[c].dtype) != "category"]

                df_values = normalization_df(x[numeric_cols])
                noisy_features: List[np.ndarray] = []
                for col in fi_list[:number_top_fi]:
                    pos_shap_feat_values = (
                        df_values[col]
                        .loc[df_shap[col].loc[df_shap[col] > 0].index.to_list()]
                        .mean()
                    )
                    neg_shap_feat_values = (
                        df_values[col]
                        .loc[df_shap[col].loc[df_shap[col] < 0].index.to_list()]
                        .mean()
                    )
                    distance_between_populations = abs(pos_shap_feat_values - neg_shap_feat_values)
                    noisy_features.append(distance_between_populations)
                return noisy_features

            def get_noisy_features_by_distance(
                distance: Union[float, int],
                list_dimension: Sequence[str],
                noisy_features: pd.DataFrame,
            ) -> pd.DataFrame:
                suspicious = noisy_features.loc[
                    noisy_features[list_dimension[0]] <= distance
                ].index.to_list()
                for i in range(1, len(list_dimension)):
                    suspicious_dim = noisy_features.loc[
                        noisy_features[list_dimension[i]] <= distance
                    ].index.to_list()
                    for row in suspicious:
                        if row not in suspicious_dim:
                            suspicious.remove(row)
                return noisy_features.filter(items=suspicious, axis=0)

            vals = get_shap_values(model, x)
            fi_list: List[str] = calculate_shap_fi(model, x)["col_name"].to_list()

            # TEMPORARY SOLUTION - ignore categorical features
            fi_list = [f for f in fi_list if str(x[f].dtype) != "category"]

            noisy_features = pd.DataFrame(fi_list[:number_top_fi], columns=["col_name"])
            if (len(np.array(vals).shape) == 2):
                noisy_features["separate_vals"] = get_noisy_features_per_dim(
                    vals, x, fi_list, number_top_fi
                )
                noisy_features.sort_values(by="separate_vals", inplace=True)
                noisy_features = noisy_features.loc[noisy_features["separate_vals"] <= distance]

            else:
                list_dimension: List[str] = []
                for i in range(len(vals)):
                    list_dimension.append(f"separate_vals_dimension_{i + 1}")
                    noisy_features[list_dimension[i]] = get_noisy_features_per_dim(
                        vals[i], x, fi_list, number_top_fi
                    )
                noisy_features.sort_values(by="separate_vals_dimension_1", inplace=True)
                noisy_features = get_noisy_features_by_distance(
                    distance, list_dimension, noisy_features
                )
                if (
                    noisy_features["separate_vals_dimension_1"]
                    == noisy_features["separate_vals_dimension_2"]
                ).all() and len(noisy_features.columns) == 3:
                    noisy_features = noisy_features.drop("separate_vals_dimension_2", axis=1)
                    noisy_features = noisy_features.rename(
                        {"separate_vals_dimension_1": "separate_vals"}, axis=1
                    )
            return noisy_features.reset_index(drop=True)

        base_model = copy.deepcopy(model)
        base_model.fit(x_train, y_train)
        final_drop_list = []
        old_score = np.around(metric(y_valid, base_model.predict(x_valid)),4)

        while True:
            base_score = old_score
            new_score = base_score
            drop_list = get_noisy_features_iteration(
                base_model, x_train, distance=distance, number_top_fi=number_top_fi
            )["col_name"].to_list()
            for col in drop_list:
                x_train_new = x_train.drop(col, axis=1)
                x_valid_new = x_valid.drop(col, axis=1)

                new_model = copy.deepcopy(base_model)
                new_model.fit(x_train_new, y_train)

                new_score = np.around(metric(y_valid, base_model.predict(x_valid)),4)
                flag = False

                if min_or_max == "max":
                    if new_score >= old_score:
                        flag = True
                elif min_or_max == "min":
                    if new_score <= old_score:
                        flag = True
                else:
                    print("min or max value not recognized")

                if flag:
                    print("Bad noisy feature found:", col)
                    print(
                        "old score:",
                        old_score,
                        "new score:",
                        new_score,
                    )
                    base_model = new_model
                    old_score = new_score
                    x_train = x_train_new
                    x_valid = x_valid_new
                    final_drop_list.append(col)
                    break

            if min_or_max == "max":
                if base_score >= new_score:
                    break

            if min_or_max == "min":
                if base_score <= new_score:
                    break

        return final_drop_list
    
def calculate_shap_fi(model, x, with_direction=False):
    """
    Calculate feature importance using SHAP values.
    - model: Trained model to analyze.
    - x: Input features DataFrame.
    - with_direction: Whether to consider the direction of the SHAP value impact.
    """
    vals = get_shap_values(model, x)
    if not with_direction:
        vals = np.abs(vals)
    if len(vals.shape) > 2: 
        vals = vals.mean(0)
    feature_importance = pd.DataFrame(list(zip(x.columns, sum(vals))), columns=['col_name', 'feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    if not with_direction:
        feature_importance['feature_importance_vals'] /= feature_importance['feature_importance_vals'].sum()
    return feature_importance.reset_index(drop=True)

def get_shap_values(model, x):
    """
    Get SHAP values for a given model and dataset.
    - model: Trained model to analyze.
    - x: Input features DataFrame.
    """
    explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
    return explainer.shap_values(x)

def get_top_fi(model, x, p):
    """
    Calculate feature importance based on accumulation of importance.
    - model: Trained model to analyze.
    - x: Input features DataFrame.
    - p: The percentile for selecting top features based on their importance.
    """
    df = get_shap_fi_and_separation_score(model, x)
    df['imp_fi'] = df['feature_importance_vals'].cumsum() < p
    return df.loc[df['imp_fi']].drop('imp_fi', axis = 1)

