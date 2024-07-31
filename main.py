from layers import Dense, Tanh
from losses import mse, mse_prime

import numpy as np



X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

neural_1 = [
    Dense(input_size= 2, output_size= 5),
    Tanh(),
    Dense(input_size= 5, output_size= 1),
    Tanh()
]

epochs = 100
learning_rate = 0.1

for epoch in range(epochs):
    error = 0
    for x, y in zip(X,Y):
        # Forward Feed
        output = x
        for layer in neural_1:
            output = layer.forward(output)

        # Error Calculation
        error += mse(y, output)

        # Backward Feed
        grad = mse_prime(y, output)
        for layer in reversed(neural_1):
            grad = layer.backward(grad, learning_rate)

    error /= len(X)
    print(f'Epoch: {epoch+1}/{epochs} | MSE: {error}')

def predict(network, val_input):
    output = val_input
    print(f'Initial Input: {output}')
    for i, layer in enumerate(network):
        output = layer.forward(output)
        print(f'layer {i}: {output}')
    
    average = sum(output[0]) / len(output[0])
    print(f'Average: {average}')
    return average

print('\n\n\n')
print(f'prediction: {predict(neural_1, [1,0])}')
print('\n\n\n')
print(f'prediction: {predict(neural_1, [0,0])}')