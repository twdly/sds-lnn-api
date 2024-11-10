import torch
import torch.nn as nn

device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


class Reservoir(nn.Module):
    def __init__(self, features, neurons, theta, tau):
        super(Reservoir, self).__init__()
        self.features = features
        self.neurons = neurons
        self.theta = theta
        self.tau = tau

        # neural ntwrk f((x_t, i_t, θ))
        self.rnn = nn.RNN((self.features + self.neurons), self.theta, batch_first=True)
        self.back2neurons = nn.Linear(self.theta, self.neurons)
        # self.activation = nn.Tanh()
        self.activation = nn.ReLU()

        self.A = nn.Parameter(torch.randn(self.neurons))

        # making everything non-trainable
        for param in self.rnn.parameters():
            param.requires_grad = False
        for param in self.back2neurons.parameters():
            param.requires_grad = False
        self.A.requires_grad = False

    def netwrkf(self, cat, rnn_state):
        output, rnn_new = self.rnn(cat, rnn_state)
        synaptic_response = self.back2neurons(output)
        # activations = self.activation(synaptic_response)
        activations = self.activation(synaptic_response)
        return activations, rnn_new

    def fused_solver(self, x, I, rnn_state, h):
        cat = torch.cat([x, I], dim=1).unsqueeze(1)  # add time dimension for batch processing because now shape B x t x (C+N)
        _f, rnn_state_new = self.netwrkf(cat, rnn_state)

        # fused solver
        numerator = x + h * (_f * self.A)  # x(t) + h * [f(.) ⊙ A]
        denominator = 1 + h * (1/self.tau + _f)  # 1 + h * (1/τ + f(.))
        x_next = (numerator / denominator).squeeze(1)  # remove t dimension because only self rnn needed it
        return x_next, rnn_state_new

    def forward(self, I, h=0.1):
        batch_size, time, features = I.shape

        # init x(0)
        x_t = torch.zeros(batch_size, self.neurons, device=device)  # shape B x N
        rnn_state = torch.zeros(1, batch_size, self.theta, device=device)

        x = []

        for t in range(time):
            I_t = I[:, t, :]  # shape B x C
            x_next, rnn_state = self.fused_solver(x_t, I_t, rnn_state, h)
            x.append(x_next)

        return torch.stack(x, dim=1)


class ReadOut(nn.Module):  # MLP read out function
    def __init__(self, features, num_of_neurons, theta, tau, output_features, lr=0.001):
        super(ReadOut, self).__init__()
        self.reservoir = Reservoir(features, num_of_neurons, theta, tau)
        self.readout = nn.Linear(num_of_neurons, output_features)
        # self.activation = nn.ReLU()
        # self.output = nn.Sigmoid()

        self.get_loss = nn.MSELoss()
        self.optimiser = torch.optim.AdamW(self.parameters(), lr=lr)

    # def normalise_tensor(self, tensor):
    #     min_val = tensor.min()
    #     max_val = tensor.max()
    #     return (tensor - min_val) / (max_val - min_val + 1e-8)

    def forward(self, I):  # get tensor function already makes I of shape B x T x F
        with torch.no_grad():
            neural_dynamics = self.reservoir(I)

        predictions = self.readout(neural_dynamics)
        # predictions = self.activation(predictions)
        # output = self.output(predictions)
        # output = self.normalise_tensor(predictions)

        return predictions, neural_dynamics

    def train_model(self, I, target, epochs=500):

        self.train()
        neural_dynamics = self.reservoir(I)

        for n in range(epochs):
            self.optimiser.zero_grad()
            predictions = self.readout(neural_dynamics)
            # predictions = self.activation(predictions)
            # output = self.output(predictions)
            # output = self.normalise_tensor(predictions)

            loss = self.get_loss(predictions, target)
            loss.backward()
            self.optimiser.step()

            if (n + 1) % epochs == 0:  # Print every 500 epochs
                print(f'Epoch {n + 1}/{epochs}, Loss: {loss.item()}')
