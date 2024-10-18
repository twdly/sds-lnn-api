from random import random

import torch

import LTCArchitecture as ltc
import Preprocessor
import pandas as pd

"""
#import other statements
after model is saved which is model.pth to load 

import LTCArchitecture as <whatever>
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

features = 10 
outputs = 2

neurons = 700
theta = 450
tau = 20

model = <whatever>.readout(features, neurons, theta, tau, outputs) #instalise model (like build it)

#load the saved weights
model.load_state_dict(torch.load('model.pth'))
model.to(device)
model.eval()

with torch.no_grad():
    predictions, _ = model.forward(I) # input I needs to be a dataframe and have the same columns as:
    ['MONTH', 'DAY', 'HOUR', 'BOM_LAT', 'BOM_LON', 'BOM_WIND', 'BOM_PRES', 'BOM_GUST', 'BOM_EYE', 'STORM_SPEED']



"""


# This function simulates receiving new input data and having the LNN making a prediction with that new data
# In the future, get_prediction() will instead create a new tensor with the data and pass it into the LNN
def update_state(month, day, hour, lat, lon, wind, pres, gust, eye, storm_speed):

    model = load_model()
    df_dict = {
        "MONTH": [month],
        "DAY": [day],
        "HOUR": [hour],
        "BOM_LAT": [lat],
        "BOM_LON": [lon],
        "BOM_WIND": [wind],
        "BOM_PRES": [pres],
        "BOM_GUST": [gust],
        "BOM_EYE": [eye],
        "STORM_SPEED": storm_speed
    }
    df = pd.DataFrame(df_dict)
    tensors = Preprocessor.get_tensors_to_predict(df)
    with torch.no_grad():
        predictions, _ = model.forward(tensors)

    predicted = predictions.squeeze(0).cpu()
    p_nature, p_intensity = predicted[:, 0], predicted[:, 1]
    file = open("state.txt", "a")
    file.write(f"{p_nature[0]},{p_intensity[0]},{month},{day},{hour},{lat},{lon},{wind},{pres},{gust},{eye},{storm_speed}\n")
    file.close()


def get_state():
    file = open("state.txt", "r")
    lines = file.readlines()
    return get_line_result(lines, len(lines) - 1)


def get_history(count):
    file = open("state.txt", "r")
    lines = file.readlines()
    line_count = len(lines)
    if line_count < count:
        count = line_count

    results = []
    i = line_count - count
    while i < line_count:
        results.append(get_line_result(lines, i))
        i += 1
    for result in results:
        result["index"] -= (line_count - count - 1)

    return results


def get_line_result(lines, index):
    current_state = lines[index].removesuffix("\n").split(",")
    result = {
        "index": index,
        "nature": float(current_state[0]),
        "intensity": float(current_state[1]),
        "month": float(current_state[2]),
        "day": float(current_state[3]),
        "hour": float(current_state[4]),
        "lat": float(current_state[5]),
        "lon": float(current_state[6]),
        "wind": float(current_state[7]),
        "pres": float(current_state[8]),
        "gust": float(current_state[9]),
        "eye": float(current_state[10]),
        "speed": float(current_state[11])
    }
    return result


def load_model():
    features = 10
    outputs = 2

    neurons = 700
    theta = 450
    tau = 20

    model = ltc.ReadOut(features, neurons, theta, tau, outputs)
    model.load_state_dict(torch.load('model.pth'))
    model.to(device)
    model.eval()
    return model


device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
