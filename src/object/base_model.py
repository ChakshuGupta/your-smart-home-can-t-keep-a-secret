import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

class BaseModel:

    def __init__(self, features):
        """
        Initialize the base model class
        """
        self.features = features
        self.process_input()


    def process_input(self):
        """
        Process the input data in the following 2 steps:
        1. Encode the destination port value using one-hot encoding
        2. Perform PCA on the endoded destination port values and get the top 
           50 most used values
        
        """
        # Use One-hot encoding to encode the destination ports
        oh_encoder = OneHotEncoder(handle_unknown='ignore')
        encoded_dport = oh_encoder.fit_transform(self.features[["dport"]])

        # Use PCA to identify the 50 most used ports
        pca = PCA(n_components=50)
        pca_encoded_dport = pca.fit_transform(encoded_dport.toarray())
        print(pca_encoded_dport)

        # # Replace the original dport column in the features with the pca encoded one
        self.features.drop(columns=["dport"], inplace=True)
        self.features = self.features.join(pd.DataFrame(pca_encoded_dport))

        print(self.features)


    def build_model(self, labels):
        """
        Build the model using Random forest classifier with
        the processed features and lables as input data and corresponding label.
        """
        self.rf_model = RandomForestClassifier()
        self.rf_model.fit(self.features, labels.values.ravel())
    
    
    def save_model(self, filepath):
        """
        Save the model as a .sav file.
        """
        pickle.dump(self.model, open(filepath, 'wb'))
        