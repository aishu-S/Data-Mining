import pandas as pd
# import numpy as np
import feature_ext_own
import pickle

T = pd.read_csv('test.csv',header=None)
T = feature_ext_own.feature_extract(T)

loaded_model = pickle.load(open('RF', 'rb'))

result = loaded_model.predict(T) # Contains an array of the predicted results

print('predicted>>',result)



