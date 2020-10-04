import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


class MyMinMaxScaler(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def trasnform(self, X, y=None):

        Xtemp = X.copy()
        scaler = MinMaxScaler()
        Xscaled = scaler.fit_transform(Xtemp)
        Xtemp = pd.DataFrame(Xscaled, columns=Xtemp.columns.to_list())


class MyStandardScaler(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def trasnform(self, X, y=None):

        Xtemp = X.copy()
        scaler = StandardScaler()
        Xscaled = scaler.fit_transform(Xtemp)
        Xtemp = pd.DataFrame(Xscaled, columns=Xtemp.columns.to_list())


class MyRobustScalerTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        Xtemp = X.copy()

        scaler = RobustScaler()
        Xscaled = scaler.fit_transform(Xtemp)
        Xtemp = pd.DataFrame(Xscaled, columns=Xtemp.columns.to_list())

        return Xtemp


class MyOneHotEncoderTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, features_name: list):
        self.features_name = features_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None) -> pd.DataFrame:

        Xtemp = X.copy()
        Xtemp = Xtemp[self.features_name]

        scaler = OneHotEncoder(drop='first')
        Xscaled = scaler.fit_transform(Xtemp)
        colunas = list(scaler.get_feature_names(self.features_name))
        Xtemp = pd.DataFrame(Xscaled.toarray(), columns=colunas)

        return Xtemp