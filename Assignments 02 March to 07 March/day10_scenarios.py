"""
Day 10 - Deep Learning Basics: Perceptron & Neural Networks (NumPy)
====================================================================
Scenarios:
  1. Student Pass/Fail Prediction (Single Perceptron)
  2. Restaurant Order Size Prediction (Single Perceptron)
  3. Customer Purchase Prediction (Multi-layer Neural Network)
  4. Online Course Completion Prediction (Neural Network)
"""

import numpy as np

# ─────────────────────────────────────────────────────────────
# SCENARIO 1: Student Pass/Fail Prediction (Perceptron)
# ─────────────────────────────────────────────────────────────
# We want to classify whether a student passes (1) or fails (0)
# based on their study hours.
# Rule: study hours >= 5 -> Pass, study hours < 5 -> Fail
# We build a single Perceptron from scratch using NumPy.
# The perceptron learns this rule by adjusting its weight and
# bias over 10 training epochs using the perceptron update rule:
#   weight = weight + learning_rate * error * input
# This is the simplest form of a neural network — one neuron.
print("=" * 60)
print("SCENARIO 1: Student Pass/Fail Prediction (Perceptron)")
print("Rule: study hours >= 5 -> Pass, < 5 -> Fail")
print("=" * 60)

np.random.seed(0)
X = np.array([[2], [4], [6], [8]])
y = np.array([0, 0, 1, 1])

weights       = np.array([0.5])
bias          = np.array([0.0])
learning_rate = 0.1

def activation(z):
    return 1 if z >= 0 else 0

for epoch in range(10):
    print(f"Epoch {epoch + 1}")
    for i in range(len(X)):
        z      = np.dot(X[i], weights) + bias
        y_pred = activation(z)
        error  = y[i] - y_pred
        weights += learning_rate * error * X[i]
        bias    += learning_rate * error
        print(f"  Input: {X[i]}, Predicted: {y_pred}, Actual: {y[i]}, Error: {error}")

print(f"\nFinal Weights: {weights}")
print(f"Final Bias: {bias}")

test_hours = np.array([3, 7])
for h in test_hours:
    z = np.dot(h, weights) + bias
    print(f"Study Hours: {h}, Prediction: {activation(z)}")

# ─────────────────────────────────────────────────────────────
# SCENARIO 2: Restaurant Order Size Prediction (Perceptron)
# ─────────────────────────────────────────────────────────────
# A restaurant wants to classify orders as Large (1) or Small (0)
# based on the number of items ordered.
# Rule: items >= 3 -> Large order, items < 3 -> Small order
# Again, a single perceptron is trained from scratch.
# The perceptron adjusts its weight each time it makes a wrong
# prediction, slowly learning the decision boundary.
print("\n" + "=" * 60)
print("SCENARIO 2: Restaurant Order Size Prediction (Perceptron)")
print("Rule: items >= 3 -> Large, < 3 -> Small")
print("=" * 60)

np.random.seed(0)
X = np.array([[1], [2], [3], [5]])
y = np.array([0, 0, 1, 1])

weights       = np.array([0.5])
bias          = np.array([0.0])
learning_rate = 0.1

for epoch in range(10):
    print(f"Epoch {epoch + 1}")
    for i in range(len(X)):
        z      = np.dot(X[i], weights) + bias
        y_pred = activation(z)
        error  = y[i] - y_pred
        weights += learning_rate * error * X[i]
        bias    += learning_rate * error
        print(f"  Items: {X[i]}, Predicted: {y_pred}, Actual: {y[i]}, Error: {error}")

print(f"\nFinal Weights: {weights}")
print(f"Final Bias: {bias}")

test_items = np.array([2, 4])
for items in test_items:
    z = np.dot(items, weights) + bias
    print(f"Items Ordered: {items}, Prediction: {activation(z)}")

# ─────────────────────────────────────────────────────────────
# SCENARIO 3: Customer Purchase Prediction (Multi-layer NN)
# ─────────────────────────────────────────────────────────────
# A single perceptron can only learn linear patterns.
# For more complex problems, we need multiple layers.
# Here we predict if a customer will buy (1) or not (0) based on:
#   - Ad Clicks (how many times they clicked an online ad)
#   - Time on Website (minutes spent browsing)
# We build a 2-layer neural network from scratch:
#   Input (2) -> Hidden Layer (2 neurons) -> Output (1 neuron)
# Sigmoid activation is used. Backpropagation updates weights.
# Trained for 1000 epochs.
print("\n" + "=" * 60)
print("SCENARIO 3: Customer Purchase Prediction (Multi-layer NN)")
print("Features: Ad Clicks, Time on Website")
print("Architecture: Input(2) -> Hidden(2) -> Output(1)")
print("=" * 60)

X = np.array([[1, 2], [2, 1], [4, 5], [5, 6]])
y = np.array([[0], [0], [1], [1]])

np.random.seed(42)
weights_input_hidden  = np.random.rand(2, 2)
bias_hidden           = np.random.rand(1, 2)
weights_hidden_output = np.random.rand(2, 1)
bias_output           = np.random.rand(1, 1)
learning_rate         = 0.1

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return z * (1 - z)

for epoch in range(1000):
    # Forward pass
    hidden_input  = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    final_input   = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output  = sigmoid(final_input)

    # Backpropagation
    error    = y - final_output
    d_output = error * sigmoid_derivative(final_output)

    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden     = error_hidden * sigmoid_derivative(hidden_output)

    # Weight updates
    weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
    bias_output           += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden  += X.T.dot(d_hidden) * learning_rate
    bias_hidden           += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

test_data   = np.array([[3, 4], [1, 1]])
hidden_test = sigmoid(np.dot(test_data, weights_input_hidden) + bias_hidden)
final_test  = sigmoid(np.dot(hidden_test, weights_hidden_output) + bias_output)
print("Predictions for test data (closer to 1 = will buy, closer to 0 = won't buy):")
print(final_test)

# ─────────────────────────────────────────────────────────────
# SCENARIO 4: Online Course Completion Prediction (NN)
# ─────────────────────────────────────────────────────────────
# An e-learning platform wants to predict if a student will
# complete an online course (1) or drop out (0) based on:
#   - Videos Watched: number of course videos the student watched
#   - Time Spent on Platform: total minutes on the platform
# A 3-layer neural network is built from scratch:
#   Input (2) -> Hidden Layer (3 neurons) -> Output (1 neuron)
# Data is normalized before training. Trained for 5000 epochs.
# This is a more complex network than Scenario 3.
print("\n" + "=" * 60)
print("SCENARIO 4: Online Course Completion Prediction (NN)")
print("Features: Videos Watched, Time Spent on Platform")
print("Architecture: Input(2) -> Hidden(3) -> Output(1)")
print("=" * 60)

X = np.array([[2, 15], [3, 20], [8, 60], [9, 75]])
y = np.array([[0], [0], [1], [1]])

X = X / np.max(X, axis=0)  # normalize

np.random.seed(1)
w1 = np.random.rand(2, 3)
w2 = np.random.rand(3, 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

for i in range(5000):
    hidden_input  = np.dot(X, w1)
    hidden_output = sigmoid(hidden_input)
    final_input   = np.dot(hidden_output, w2)
    output        = sigmoid(final_input)

    error    = y - output
    d_output = error * sigmoid_derivative(output)

    error_hidden = d_output.dot(w2.T)
    d_hidden     = error_hidden * sigmoid_derivative(hidden_output)

    w2 += hidden_output.T.dot(d_output)
    w1 += X.T.dot(d_hidden)

print("Final Predictions (closer to 1 = will complete, closer to 0 = will drop out):")
print(output)

print("\n" + "=" * 60)
print("DAY 10 - ALL SCENARIOS COMPLETE")
print("=" * 60)
