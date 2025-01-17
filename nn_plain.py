import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.cm import get_cmap

class nn_plain():
    # Initialize neural network parameters
    input_size = 1
    hidden_size1 = 8
    hidden_size2 = 8
    output_size = 1
    
    # Random initialization of weights and biases
    np.random.seed(42)

    
    
    def __init__(self, hidden_size1 = 8, hidden_size2 = 8):
        """
        Initializes the neural network parameters: weights, biases, and epoch counter.
    
        - Weights are initialized with small random values to break symmetry and allow the network to learn.
        - Biases are initialized to zero.
        - The epoch counter is initialized to zero.
    
        Attributes:
            W1 (numpy.ndarray): Weights for the connections between input layer and first hidden layer.
            b1 (numpy.ndarray): Biases for the first hidden layer.
            W2 (numpy.ndarray): Weights for the connections between first hidden layer and second hidden layer.
            b2 (numpy.ndarray): Biases for the second hidden layer.
            W3 (numpy.ndarray): Weights for the connections between second hidden layer and output layer.
            b3 (numpy.ndarray): Biases for the output layer.
            epoch (int): Counter to track the number of epochs during training.
        """
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        
        # Initialize weights for the input layer to the first hidden layer
        self.W1 = np.random.randn(self.input_size, self.hidden_size1) * 0.1  # Small random values
        self.b1 = np.zeros((1, self.hidden_size1))  # Biases initialized to zero
    
        # Initialize weights for the first hidden layer to the second hidden layer
        self.W2 = np.random.randn(self.hidden_size1, self.hidden_size2) * 0.1
        self.b2 = np.zeros((1, self.hidden_size2))
    
        # Initialize weights for the second hidden layer to the output layer
        self.W3 = np.random.randn(self.hidden_size2, self.output_size) * 0.1
        self.b3 = np.zeros((1, self.output_size))
    
        # Initialize epoch counter to zero
        self.epoch = 0
            
    # Activation functions
    def relu(self, x):
        # return np.maximum(0, x)
        # return np.maximum(0, np.exp(x))
        return np.sin(2*np.pi*x)
    
    def relu_derivative(self, x):
        # return (x > 0).astype(float)
        return np.cos(2*np.pi*x)
    
  
    
    def forward(self, x):
        """
        Performs a forward pass through the neural network.
        
        Args:
            x (numpy.ndarray): Input data of shape (batch_size, input_size).
        
        Returns:
            numpy.ndarray: Output of the neural network of shape (batch_size, output_size).
        """
        # First layer: Compute the weighted sum of inputs and add bias
        self.z1 = np.dot(x, self.W1) + self.b1  # z1: Pre-activation of first hidden layer
        self.a1 = self.relu(self.z1)           # a1: Activation of first hidden layer
    
        # Second layer: Compute the weighted sum from the first hidden layer and add bias
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # z2: Pre-activation of second hidden layer
        self.a2 = self.relu(self.z2)                 # a2: Activation of second hidden layer
    
        # Output layer: Compute the weighted sum from the second hidden layer and add bias
        self.z3 = np.dot(self.a2, self.W3) + self.b3  # z3: Pre-activation of output layer
        self.output = self.z3                        # Linear output, no activation for regression tasks
    
        # Return the final output of the network
        return self.output
        

    
    
    def backward(self, x, y, learning_rate=0.01):
        """
        Performs a backward pass through the neural network (backpropagation).
        Computes the gradients of the loss with respect to the weights and biases,
        and updates the parameters using gradient descent.
    
        Args:
            x (numpy.ndarray): Input data of shape (batch_size, input_size).
            y (numpy.ndarray): Target values of shape (batch_size, output_size).
            learning_rate (float): Learning rate for gradient descent updates.
    
        Returns:
            None
        """
        # Number of training examples in the batch
        m = x.shape[0]
    
        # Compute the gradient of the loss with respect to the output
        # Loss = Mean Squared Error: (1/m) * sum((output - y)^2)
        # Derivative of loss w.r.t. output = 2 * (output - y) / m
        d_output = (self.output - y) / m
    
        # Gradients for the output layer (W3 and b3)
        dW3 = np.dot(self.a2.T, d_output)              # Gradient of W3
        db3 = np.sum(d_output, axis=0, keepdims=True)  # Gradient of b3
    
        # Backpropagate through the second hidden layer
        dA2 = np.dot(d_output, self.W3.T)              # Gradient w.r.t. activations in layer 2
        dZ2 = dA2 * self.relu_derivative(self.z2)      # Apply derivative of ReLU
        dW2 = np.dot(self.a1.T, dZ2)                   # Gradient of W2
        db2 = np.sum(dZ2, axis=0, keepdims=True)       # Gradient of b2
    
        # Backpropagate through the first hidden layer
        dA1 = np.dot(dZ2, self.W2.T)                   # Gradient w.r.t. activations in layer 1
        dZ1 = dA1 * self.relu_derivative(self.z1)      # Apply derivative of ReLU
        dW1 = np.dot(x.T, dZ1)                         # Gradient of W1
        db1 = np.sum(dZ1, axis=0, keepdims=True)       # Gradient of b1
    
        # Update weights and biases using gradient descent
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

        
    def train(self, x_train, y_train, epochs=1000):
        """
        Trains the neural network using the provided training data.
    
        Args:
            x_train (numpy.ndarray): Input training data of shape (num_samples, input_size).
            y_train (numpy.ndarray): Target training data of shape (num_samples, output_size).
            epochs (int): Number of epochs to train the model.
    
        Returns:
            None
        """
        # Set the learning rate
        learning_rate = 0.1
    
        # Perform a forward pass on the untrained network to calculate the initial loss
        output = self.forward(x_train)  # Predictions
        self.loss = np.mean((output - y_train) ** 2)  # Initial loss
        self.x_train = x_train
        self.y_train = y_train
        print(f"Epoch {self.epoch}, Loss: {self.loss:.6f}")  # Print the loss for epoch 0
    
        # Training loop
        for i in range(epochs):
            self.epoch += 1  # Increment the epoch count
    
            # Forward pass: Compute predictions
            output = self.forward(x_train)
    
            # Compute loss: Mean squared error
            self.loss = np.mean((output - y_train) ** 2)
    
            # Backward pass: Update weights and biases using gradient descent
            self.backward(x_train, y_train, learning_rate)
    
            # Print the loss every 500 epochs for monitoring
            # if self.epoch % 500 == 0:
            #     print(f"Epoch {self.epoch}, Loss: {self.loss:.6f}")
    
        # Print the final loss after training
        print(f"Epoch {self.epoch}, Loss: {self.loss:.6f}")
            
        
    def visualize_network(self, axes, plotLabels=True):
        """
        Visualizes the structure of the neural network with weights and biases.
    
        Args:
            axes (matplotlib.axes.Axes): Matplotlib axes object to plot the network.
            plotLabels (bool): If True, display weight values as labels on the edges.
    
        Returns:
            None
        """
        # Define the network structure
        weights = [self.W1, self.W2, self.W3]  # List of weight matrices for each layer
        biases = [self.b1, self.b2, self.b3]  # List of bias vectors for each layer
        layer_sizes = [self.W1.shape[0], self.W1.shape[1], self.W2.shape[1], self.W3.shape[1]]  # Number of neurons per layer
    
        # Create a directed graph using NetworkX
        G = nx.DiGraph()
    
        # Create positions for the nodes in the graph (neurons)
        pos = {}  # Dictionary to store node positions
        y_spacing = 1.5  # Spacing between neurons in the same layer
        x_spacing = 5    # Spacing between layers
    
        nodes_by_layer = []  # Store nodes layer-by-layer for easy reference
    
        for layer_idx, size in enumerate(layer_sizes):
            layer_nodes = []
            for neuron_idx in range(size):
                # Generate a unique node name for each neuron (e.g., L0_N0 for layer 0, neuron 0)
                node_name = f"L{layer_idx}_N{neuron_idx}"
                layer_nodes.append(node_name)
    
                # Position the neuron in the graph
                pos[node_name] = (
                    layer_idx * x_spacing,  # X-coordinate based on the layer
                    neuron_idx * y_spacing - (size - 1) * y_spacing / 2,  # Y-coordinate centered vertically
                )
    
                # Add the neuron as a node in the graph
                G.add_node(node_name)
    
            nodes_by_layer.append(layer_nodes)
    
        # Add edges with weights and biases to the graph
        cmap = get_cmap("coolwarm")  # Colormap for visualizing biases
        edge_widths = []  # List to store edge widths for visualization
        edge_colors = []  # List to store edge colors based on bias values
        edge_labels = {}  # Dictionary to store edge labels (weights)
    
        for layer_idx, (W, b) in enumerate(zip(weights, biases)):
            for i in range(W.shape[0]):  # From neurons in the current layer
                for j in range(W.shape[1]):  # To neurons in the next layer
                    # Get source and target nodes
                    source = nodes_by_layer[layer_idx][i]
                    target = nodes_by_layer[layer_idx + 1][j]
    
                    # Extract weight and bias
                    weight = W[i, j]
                    bias = b[0, j] if b.ndim > 1 else b[j]
    
                    # Add edge to the graph
                    G.add_edge(source, target, weight=weight, bias=bias)
    
                    # Style the edges: width by weight magnitude, color by bias value
                    edge_widths.append(abs(weight))  # Edge width proportional to weight magnitude
                    edge_colors.append(cmap((bias + 1) / 2))  # Normalize bias to [0, 1]
    
                    # Add weight label
                    edge_labels[(source, target)] = f"{weight:.2f}"
    
        # Normalize edge widths for better visualization
        max_width = max(edge_widths) if edge_widths else 1
        edge_widths = [w / max_width * 3 for w in edge_widths]
    
        # Draw the graph using NetworkX
        nx.draw(
            G,
            pos,  # Node positions
            with_labels=False,  # Do not label nodes
            node_color="skyblue",  # Node color
            node_size=700,  # Node size
            edge_color=edge_colors,  # Edge colors based on biases
            width=edge_widths,  # Edge widths based on weights
            arrows=False,  # No arrows for edges
        )
    
        # Add edge labels for weights if requested
        if plotLabels:
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels=edge_labels, font_color="black", font_size=8
            )
    
        # Set the title and remove axes
        axes.set_title(f"Epoch: {self.epoch}, Loss: {self.loss:.6f}")
        axes.axis("off")




    def test_plot(self, axes, title = ''):
        # Generate test data
        # x_test = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

        xmin = min(self.x_train)
        xmax = max(self.x_train)
        x_test = np.linspace(xmin, xmax, 100) + np.random.uniform(-0.01, 0.01, 100)
        # x_test /= max(x_test)
        x_test = x_test.reshape(-1, 1)
        # x_test = np.random.rand(100).reshape(-1, 1)
        # x_test = np.random.uniform(0, 1, 100).reshape(-1, 1)
        # x_test_normalized = x_test / (2 * np.pi)
        # y_test = np.sin(x_test*2*np.pi)
        y_test = self.y_train
        
        
        # Predict using the trained model
        y_pred = self.forward(x_test)
    
        # Plot the results
        # axes.plot(x_test, y_test, label='True Sine Function', color='blue')
        axes.scatter(x_test, y_pred, label='Predicted Values', color='red')
        axes.scatter(self.x_train, self.y_train, label='Training Data', color='green', alpha = 0.7,edgecolors='none')
        
        axes.set_ylim(min(y_test)-0.5, max(y_test)+0.5 )
        axes.set_title(title)
        axes.set_xlabel('x')
        axes.set_ylabel('sin(x)')
        
        axes.legend()
        axes.grid()
        # plt.show()        
