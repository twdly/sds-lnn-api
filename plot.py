import numpy as np
import plotly.graph_objects as go
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_predictions(p, t, year):
    predicted, target = p.squeeze(0).cpu(), t.squeeze(0).cpu()

    p_nature, p_intensity = predicted[:, 0], predicted[:, 1]  # N I
    t_nature, t_intensity = target[:, 0], target[:, 1]

    time = np.arange(len(p_nature))

    predFig = go.Figure()

    predFig.add_trace(go.Scatter(
        x=time,
        y=t_nature,
        mode='lines',
        name='True Nature',
        line=dict(color='green')
    ))

    predFig.add_trace(go.Scatter(
        x=time,
        y=p_nature,
        mode='lines',
        name='Predicted Nature',
        line=dict(color='red')
    ))

    predFig.add_trace(go.Scatter(
        x=time,
        y=t_intensity,
        mode='lines',
        name='True Intensity',
        line=dict(color='orange')
    ))

    predFig.add_trace(go.Scatter(
        x=time,
        y=p_intensity,
        mode='lines',
        name='Predicted Intensity',
        line=dict(color='blue')
    ))

    predFig.show()


def plot_neural_dynamics(reservoir_states, year):
    neuron_states = reservoir_states.squeeze(0).cpu().numpy()
    time = np.arange(neuron_states.shape[0])
    fig = go.Figure()

    for i in range(neuron_states.shape[1]):
        fig.add_trace(go.Scatter(x=time, y=neuron_states[:, i], mode='lines', name=f'Neuron {i}'))

    fig.update_layout(
        title=f"Neural Dynamics Over Time {year}",
        xaxis_title="Time",
        yaxis_title="Neuron Activation",
        # yaxis=dict(range=[1, -1]), #it seems that 1 and -1 are too large of range for activation
        legend_title="Neurons",
        hovermode="x unified"
    )

    fig.show()