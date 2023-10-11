import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def prepare_data(data):
    le = LabelEncoder()
    data['item_type'] = le.fit_transform(data['item_type'])
    data['item_color'] = le.fit_transform(data['item_color'])
    data['item_size'] = le.fit_transform(data['item_size'])
    return data

def train_model(data):
    X = data.drop('user_preference', axis=1)
    y = data['user_preference']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def predict(clf, input_data):
    prediction = clf.predict(input_data)
    return prediction