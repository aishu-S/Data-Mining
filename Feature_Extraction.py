import pandas as pd
import numpy as np
from scipy import fftpack

MealFeatureMatrix = pd.DataFrame()
NoMealFeatureMatrix = pd.DataFrame()

def get_velocity(data, FeatureMatrix):
    print("Extracting Velocity ...")
    glucoseVelocity = pd.DataFrame()

    for i in range(0, 24):
        glucoseVelocity['velocity' + str(i)] = ((data.iloc[:, i + 1] - data.iloc[:, i]) / 5)

    FeatureMatrix['meanVelocity'] = glucoseVelocity.mean(axis=1, skipna=True)
    print("... Finished Extracting Velocity")

def get_glucose_level_diff(data, FeatureMatrix):
    print("Extracting Glucose Level Diff ...")
    FeatureMatrix['glucoseLevelDiff'] = data.apply(lambda x: np.max(x) - np.min(x), axis=1)
    print("... Finished Extracting Glucose Level Diff")

def calculateFastFourierTransform(row):
    rowFFTValue = abs(fftpack.fft(list(row.array)))
    rowFFTValue.sort()
    return np.flip(rowFFTValue)[0:6]

def getFastFourierTransform(data):
    print("Extracting FFT ...")
    FFT = pd.DataFrame()
    FFT['vals'] = data.apply(calculateFastFourierTransform, axis=1)
    return pd.DataFrame(FFT.vals.tolist(), columns=['1', '2', '3', '4', '5', '6'])

def main():
    MealFeatureMatrix = pd.DataFrame()
    data = pd.read_csv("meal.csv")
    get_velocity(data, MealFeatureMatrix)
    get_glucose_level_diff(data, MealFeatureMatrix)
    MealFeatureMatrix = pd.concat([MealFeatureMatrix, getFastFourierTransform(data)], axis=1, ignore_index=True)
    print("... Finished Extracting FFT")
    # print(MealFeatureMatrix.head())
    np_arr = MealFeatureMatrix.to_numpy()
    # MealFeatureMatrix.to_csv("mealDataFeatures.csv", header=False, index=False)
    np.savetxt("mealDataFeatures.csv", np_arr, delimiter=",")

