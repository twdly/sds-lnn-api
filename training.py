import pandas as pd
import torch

import LTCArchitecture as ltc
import Preprocessor as p
import plot as plt

device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

df_south = pd.read_csv('SP.csv')

training_inputs, targets = p.get_training_data(p.preprocess_data(df_south), '1973', '2024')
features = training_inputs[0].shape[2]
outputs = targets[0].shape[2]
neurons = 700
theta = 450
tau = 20
lr = 0.5
epochs = 800

LTC = ltc.ReadOut(features, neurons, theta, tau, outputs, lr)
LTC.to(device)

for input, target in zip(training_inputs, targets):
    LTC.train_model(input, target, epochs)

LTC.eval()

f_20s, t_20s = p.get_tensors_to_predict_for_training(p.preprocess_data(df_south), '1990', '1990')

with torch.no_grad():
    predictions, neural_dynamics = LTC.forward(f_20s)

print(predictions.shape)

plt.plot_neural_dynamics(neural_dynamics, '1990-1990')
plt.plot_predictions(predictions, t_20s, '1990-1990')


torch.save(LTC.state_dict(), 'model.pth')