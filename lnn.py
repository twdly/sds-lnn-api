from random import random


def get_prediction():
    return random() * 100


def update_state(vorticity, temperature):
    file = open("state.txt", "a")
    file.write(f"\n{vorticity},{temperature}")
    file.close()


def get_state():
    file = open("state.txt", "r")
    lines = file.readlines()
    current_state = lines[len(lines) - 1]
    return current_state.split(",")
