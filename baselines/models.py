import torch


class FeedForwardNN(torch.nn.Module):
    """
    Simple FeedForward Neural Network. All Hidden Layers are the same size
    and we use ReLU activations
    """

    def __init__(
        self,
        input_sz: int,
        labels: int,
        hidden_sz: int = 256,
        hidden_layers: int = 2,
        dropout: float = None,
    ):
        """
        Initialize the network layers given the input hyperparams.
        :param input_sz: Size of input feature
        :param labels: Number of classes
        :param hidden_sz: Size of hidden layer (Linear)
        :param hidden_layers: Number of hidden Linear layers
        :param dropout: % of Dropout
        """
        super(FeedForwardNN, self).__init__()
        # Set up initial hidden layer with Linear and ReLU
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(input_sz, hidden_sz), torch.nn.ReLU()]
        )
        # Add dropout if specified, default is to not
        if dropout is not None:
            self.layers.append(torch.nn.Dropout(dropout))

        # If more than 1 hidden layer, repeat for all hidden layers
        for i in range(1, hidden_layers - 1):
            self.layers.append(torch.nn.Linear(hidden_sz, hidden_sz))
            if dropout is not None:
                self.layers.append(torch.nn.Dropout(dropout))
        # Add Output layer
        self.layers.extend([torch.nn.Linear(hidden_sz, labels)])

    def forward(self, x):
        # Forward pass of linear net
        for layer in self.layers:
            x = layer(x)
        return x


class VisionLSTM(torch.nn.Module):
    """
    Simple uni-directional LSTM with 1 output layer to project hidden states to label space
    """

    def __init__(
        self,
        input_sz: int,
        labels: int,
        hidden_sz: int = 256,
        hidden_layers: int = 1,
        dropout: float = None,
    ):
        """
        Initialize the network layers given the input hyperparams.
        :param input_sz: Size of input feature
        :param labels: Number of classes
        :param hidden_sz: Size of hidden layer (Size of LSTM hidden state in this case)
        :param hidden_layers: Number of LSTM layers
        :param dropout: % of Dropout (if > 0, should be a multilayer LSTM)
        """
        super(VisionLSTM, self).__init__()
        if dropout is not None:
            self.dropout = dropout
        else:
            self.dropout = 0.0
        self.lstm = torch.nn.LSTM(
            input_size=input_sz,
            hidden_size=hidden_sz,
            num_layers=hidden_layers,
            batch_first=True,
            dropout=self.dropout,
        )
        self.classifier = torch.nn.Linear(in_features=hidden_sz, out_features=labels)

    def forward(self, x):
        # Pass through LSTM and then output layer
        out, _ = self.lstm(x)
        return self.classifier(out)
