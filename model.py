import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights randomly
        self.input_to_hidden_weights = np.random.randn(input_size, hidden_size)
        self.hidden_to_output_weights = np.random.randn(hidden_size, output_size)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def feedforward(self, inputs):
        # Input to hidden layer
        hidden_sum = np.dot(inputs, self.input_to_hidden_weights)
        hidden_output = self.sigmoid(hidden_sum)
        
        # Hidden to output layer
        output_sum = np.dot(hidden_output, self.hidden_to_output_weights)
        output = self.sigmoid(output_sum)
        
        return output
    
    def train(self, inputs, targets, epochs):
        for epoch in range(epochs):
            for i in range(len(inputs)):
                # Feedforward
                input_data = inputs[i]
                target = targets[i]
                
                hidden_sum = np.dot(input_data, self.input_to_hidden_weights)
                hidden_output = self.sigmoid(hidden_sum)
                
                output_sum = np.dot(hidden_output, self.hidden_to_output_weights)
                output = self.sigmoid(output_sum)
                
                # Calculate error
                error = target - output
                
                # Update weights (no backpropagation)
                self.input_to_hidden_weights += np.outer(input_data, hidden_output) * error
                self.hidden_to_output_weights += np.outer(hidden_output, output) * error
                
                # Print error
                print(f'Epoch {epoch + 1}, Sample {i + 1}, Error: {np.mean(np.abs(error))}')

# Example usage
inputs = np.array([[0, 0, 1, 1],
                   [0, 1, 1, 0],
                   [1, 0, 1, 1],
                   [1, 1, 1, 0]])
targets = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork(input_size=4, hidden_size=3, output_size=1)
nn.train(inputs, targets, epochs=100)