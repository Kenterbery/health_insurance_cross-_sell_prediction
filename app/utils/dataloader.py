import re

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()

    def get_title(self, name):
        pattern = "([A-Za-z]+\."
        title_search = re.search(pattern, name)
        if title_search:
            return title_search.group(1)
        return ""

    def load_data(self):
        #label encode
        label_encoder = LabelEncoder()
        self.dataset["Gender"] = label_encoder.fit_transform(self.dataset["Gender"])
        self.dataset["Vehicle_Age"] = label_encoder.fit_transform(self.dataset["Vehicle_Age"])
        self.dataset["Vehicle_Damage"] = label_encoder.fit_transform(self.dataset["Vehicle_Damage"])

        #Standarn scaling
        # scaler = StandardScaler()
        # self.dataset["Vintage"] = scaler.fit_transform(self.dataset["Vintage"].to_numpy().reshape(-1,1))
        # self.dataset["Annual_Premium"] = scaler.fit_transform(self.dataset["Annual_Premium"].to_numpy().reshape(-1,1))

        return self.dataset