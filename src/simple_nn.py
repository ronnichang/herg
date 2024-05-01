import torch
import torch.nn as nn
import torch.optim as optim


class SimpleNN(nn.Module):
    def __init__(self, input_features, hidden_units, output_features, learning_rate=0.001):
        super(SimpleNN, self).__init__()
        # Define the network layers
        self.layers = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_features)
        )
        
        # Initialize the optimizer and loss function within the model
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()  # replace with appropriate loss function based on your task

    def forward(self, x):
        return self.layers(x)

    def train_model(self, X_train, y_train, epochs=200):
        self.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self(X_train)
            loss = self.loss_func(outputs, y_train)
            loss.backward()
            self.optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    def predict(self, X):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            outputs = self(X)
        return outputs

    
# Example usage:
# Define the dimensions for input, hidden layer, and output
input_features = 10  # number of input features
hidden_units = 20   # number of hidden units
output_features = 1  # number of output features

# Assuming X_train, y_train, and X_test are properly defined and converted to tensors
nn_model = SimpleNN(input_features, hidden_units, output_features)
nn_model.train_model(X_train, y_train)  # Train the model
predictions = nn_model.predict(X_test)  # Make predictions on new data
