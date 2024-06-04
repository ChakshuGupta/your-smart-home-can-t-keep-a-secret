import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

class BaseModel:

    def __init__(self, features):
        """
        """
        self.process_input(features)


    def process_input(self, features):
        """
        """
        # Use One-hot encoding to encode the destination ports
        oh_encoder = OneHotEncoder(handle_unknown='ignore')
        encoded_dport = oh_encoder.fit_transform(features[["dport"]])

        # Use PCA to identify the 50 most used ports
        pca = PCA(n_components=50)
        pca_encoded_dport = pca.fit_transform(encoded_dport.toarray())
        print(pca_encoded_dport)

        # # Replace the original dport column in the features with the pca encoded one
        features.drop(columns=["dport"], inplace=True)
        features = features.join(pd.DataFrame(pca_encoded_dport))

        print(features)


    def build_model(self, features, labels):
        rf_model = RandomForestClassifier()
        rf_model.fit(features, labels.values.ravel())
        return rf_model