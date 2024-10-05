from random import random


# This function simulates receiving new input data and having the LNN making a prediction with that new data
# In the future, get_prediction() will instead create a new tensor with the data and pass it into the LNN
def update_state(vorticity, temperature):

    def get_prediction():
        return random() * 100

    file = open("state.txt", "a")
    file.write(f"{round(get_prediction(), 2)},{vorticity},{temperature}\n")
    file.close()


def get_state():
    file = open("state.txt", "r")
    lines = file.readlines()
    current_state = lines[len(lines) - 1].removesuffix("\n").split(",")
    result = {
        "prediction": float(current_state[0]),
        "vorticity": int(current_state[1]),
        "temperature": int(current_state[2])
    }
    return result
