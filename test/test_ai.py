import torch
import pytest
import LTCArchitecture as ltc  # Ensure this is your correct module import

def test_model_training():
    features = 10
    outputs = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_tensor = torch.rand((1, 10, features), device=device)
    target_tensor = torch.rand((1, 10, outputs), device=device)

    model = ltc.ReadOut(features, 700, 450, 20, outputs, 0.5).to(device)
    initial_state_dict = {name: param.clone() for name, param in model.named_parameters()}

    model.train_model(input_tensor, target_tensor, epochs=1)

    for name, param in model.named_parameters():
        if "rnn" in name or "back2neurons" in name:
            assert param.requires_grad is False, f"{name} should not be trainable."
        elif param.grad is not None:
            grad_norm = param.grad.norm().item()
            weight_change = torch.norm(param - initial_state_dict[name]).item()
            assert grad_norm > 0, f"No gradient for {name}; grad_norm={grad_norm}"
            assert weight_change > 0, f"No change in weights for {name}; weight_change={weight_change}"
        else:
            assert False, f"{name} did not receive any gradients."

def test_model_predictions():
    features = 10  # This should be set to the number of input features your model expects
    outputs = 2    # This should be set to the number of output features your model predicts
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup a test input tensor
    test_tensor = torch.rand((1, 10, features), device=device)  # Correctly using 'features' as an integer
    
    # Initialize your model (assuming you have it imported and configured)
    model = ltc.ReadOut(features, 700, 450, 20, outputs, 0.5).to(device)
    
    # Forward pass through the model
    predictions, neural_dynamics = model.forward(test_tensor)
    
    # Example assertion: Check if the predictions have the expected shape
    assert predictions.shape == (1, 10, outputs), f"Expected shape (1, 10, {outputs}), but got {predictions.shape}"

# def test_single_sample_forward_pass():
#     features = 10
#     outputs = 2
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     test_tensor = torch.rand((1, 10, features), device=device)
#     model = ltc.ReadOut(features, 700, 450, 20, outputs, 0.5).to(device)

#     predictions, neural_dynamics = model.forward(test_tensor)
#     assert predictions.shape == (1, 10, outputs), f"Expected shape (1, 10, {outputs}), but got {predictions.shape}"

#     features = 10
#     outputs = 2
#     batch_size = 5
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     test_tensor = torch.rand((batch_size, 10, features), device=device)
#     model = ltc.ReadOut(features, 700, 450, 20, outputs, 0.5).to(device)

#     predictions, neural_dynamics = model.forward(test_tensor)
#     assert predictions.shape == (batch_size, 10, outputs), f"Expected shape ({batch_size}, 10, {outputs}), but got {predictions.shape}"

def test_verify_output_ranges():
    features = 10
    outputs = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_tensor = torch.rand((1, 10, features), device=device)
    model = ltc.ReadOut(features, 700, 450, 20, outputs, 0.5).to(device)

    predictions, neural_dynamics = model.forward(test_tensor)
    latitudes = predictions[..., 0]
    longitudes = predictions[..., 1]

    assert torch.all(latitudes >= -90) and torch.all(latitudes <= 90), "Latitude values out of range"
    assert torch.all(longitudes >= -180) and torch.all(longitudes <= 180), "Longitude values out of range"

def test_impact_of_tau_on_forward_pass():
    features = 10
    outputs = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_tensor = torch.rand((1, 10, features), device=device)
    tau_values = [0.1, 0.5, 1.0]

    for tau in tau_values:
        model = ltc.ReadOut(features, 700, 450, 20, outputs, tau).to(device)
        predictions, neural_dynamics = model.forward(test_tensor)
        assert predictions.shape == (1, 10, outputs), f"Expected shape (1, 10, {outputs}), but got {predictions.shape}" 

def test_LNN_specific_parameters():
    features = 10
    outputs = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_tensor = torch.rand((1, 10, features), device=device)
    model = ltc.ReadOut(features, 700, 450, 20, outputs, 0.5).to(device)
    predictions, neural_dynamics = model.forward(test_tensor)

    assert model.reservoir.tau == 20, "RNN tau value not set correctly"


import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_input_perturbation():
    features = 10
    outputs = 2
    batch_size = 1
    seq_length = 10
    perturbation_scale = 1e-6

    # Initialize the model
    model = ltc.ReadOut(features, 700, 450, 20, outputs, 0.5).to(device)

    # Generate a base input tensor
    base_input = torch.rand((batch_size, seq_length, features), device=device)

    # Perturb the input slightly
    perturbed_input = base_input + torch.randn_like(base_input) * perturbation_scale

    # Get predictions for both base and perturbed inputs
    base_output, _ = model(base_input)
    perturbed_output, _ = model(perturbed_input)

    # Calculate the difference between the outputs
    output_difference = torch.abs(base_output - perturbed_output).max()

    # Assert that the difference is small
    assert output_difference.item() < 1e-5, f"Model is too sensitive to input perturbations: {output_difference.item()}"

# Call the test function
test_input_perturbation()



def test_long_term_stability():
    features = 10
    outputs = 2
    batch_size = 1
    long_seq_length = 100  # Significantly longer than typical sequences

    # Initialize the model
    model = ltc.ReadOut(features, 700, 450, 20, outputs, 0.5).to(device)

    # Generate a long input tensor
    long_input = torch.rand((batch_size, long_seq_length, features), device=device)

    # Get predictions for the long input
    long_output, _ = model(long_input)

    # Check for stability:
    # Ensure that outputs are not exploding or vanishing by checking their range
    max_output = torch.max(long_output)
    min_output = torch.min(long_output)

    # Define acceptable output range thresholds
    acceptable_min, acceptable_max = -10, 10

    # Assert that the outputs are within an expected range
    assert min_output.item() > acceptable_min, f"Model output is too low, indicating instability: {min_output.item()}"
    assert max_output.item() < acceptable_max, f"Model output is too high, indicating instability: {max_output.item()}"

def test_extreme_scenarios():
    features = 10
    outputs = 2
    batch_size = 1
    seq_length = 10

    # Initialize the model
    model = ltc.ReadOut(features, 700, 450, 20, outputs, 0.5).to(device)

    # Generate extreme but realistic inputs
    # Example: very high values for temperature, pressure, etc.
    # Adjust the range based on the expected input features of your model
    extreme_input = torch.rand((batch_size, seq_length, features), device=device) * 10  # Scale to extreme values

    # Get predictions for the extreme input
    extreme_output, _ = model(extreme_input)

    # Check if the outputs are sensible
    # We assume the model should still output within a known range
    max_output = torch.max(extreme_output)
    min_output = torch.min(extreme_output)

    # Define acceptable output range thresholds for extreme scenarios
    acceptable_min, acceptable_max = -10, 10

    # Assert that the outputs are within an expected range
    assert min_output.item() > acceptable_min, f"Model output too low under extreme conditions: {min_output.item()}"
    assert max_output.item() < acceptable_max, f"Model output too high under extreme conditions: {max_output.item()}"

def test_temporal_consistency():
    features = 10
    outputs = 2
    neurons = 700
    theta = 450
    tau = 20
    batch_size = 1
    seq_length = 50  # Longer sequence for testing temporal consistency
    learning_rate = 0.001

    # Initialize the model
    model = ltc.ReadOut(features, neurons, theta, tau, outputs, learning_rate).to(device)

    # Generate a smooth test input (e.g., a sine wave)
    time = torch.linspace(0, 10, steps=seq_length).unsqueeze(0).unsqueeze(2)  # Shape: (1, seq_length, 1)
    test_input = torch.sin(time).expand(-1, -1, features).to(device)  # Shape: (1, seq_length, features)

    # Forward pass
    with torch.no_grad():
        predictions, _ = model(test_input)

    # Check if consecutive predictions are consistent
    # Calculate differences between consecutive predictions
    diff = torch.abs(predictions[:, 1:] - predictions[:, :-1])

    # Set a threshold for maximum allowed difference
    max_allowed_diff = 0.1  # This threshold should be set based on your specific requirements

    # Assert that all differences are below the threshold
    assert torch.all(diff < max_allowed_diff), "Found large differences between consecutive predictions."

def test_spatial_consistency():
    features = 10  # This might include latitude, longitude, and other related features
    outputs = 2    # Assuming outputs include latitude and longitude
    neurons = 700
    theta = 450
    tau = 20
    batch_size = 1
    seq_length = 30  # Adequate sequence length to check transitions
    learning_rate = 0.001

    # Initialize the model
    model = ltc.ReadOut(features, neurons, theta, tau, outputs, learning_rate).to(device)

    # Generate a test input where spatial positions change gradually
    # Example: Simulated trajectory of a cyclone
    lat_initial = 10.0
    lon_initial = -20.0
    lat_changes = torch.linspace(0, 1, seq_length).unsqueeze(0).unsqueeze(2)  # Gradual changes in latitude
    lon_changes = torch.linspace(0, 1, seq_length).unsqueeze(0).unsqueeze(2)  # Gradual changes in longitude
    other_features = torch.zeros((batch_size, seq_length, features - 2))  # Other features remain constant
    test_input = torch.cat([lat_changes, lon_changes, other_features], dim=2).to(device)

    # Forward pass
    with torch.no_grad():
        predictions, _ = model(test_input)

    # Extract spatial outputs (assuming they are the first two outputs)
    lat_predictions = predictions[:, :, 0]
    lon_predictions = predictions[:, :, 1]

    # Calculate differences between consecutive positions
    lat_diff = torch.abs(lat_predictions[:, 1:] - lat_predictions[:, :-1])
    lon_diff = torch.abs(lon_predictions[:, 1:] - lon_predictions[:, :-1])

    # Set thresholds for maximum allowed changes
    max_lat_change = 0.05  # This should be based on realistic cyclone movement per timestep
    max_lon_change = 0.05

    # Assert that all changes are below the thresholds
    assert torch.all(lat_diff < max_lat_change), "Unrealistic jumps in latitude predictions."
    assert torch.all(lon_diff < max_lon_change), "Unrealistic jumps in longitude predictions."