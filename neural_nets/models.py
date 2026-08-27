import torch
import torch.nn as nn

class ThermalLSTM(nn.Module):

    # TODO: variable input sequence length
    def __init__(self, hidden_dim, input_sequence_length, output_sequence_length, input_size = 1, num_layers = 1):

        super(ThermalLSTM, self).__init__()

        self.hidden_dim = hidden_dim # length of hidden state vectors in LSTM cell
        self.input_size = input_size # shape of a single timestamp's data in input sequence (always 1 for 1D timeseres)

        self.input_sequence_length  = input_sequence_length # number of points in input sequence
        self.output_sequence_length = output_sequence_length # number of predicted points in output sequence

        # LSTM cell definition
        self.lstm = nn.LSTM(input_size,
                            hidden_dim,
                            num_layers = num_layers,
                            batch_first=True, # formatting thing. changes required dimensions of lstm cell's input
                            dtype = torch.float32)

        self.head = nn.Linear(hidden_dim, output_sequence_length) # simple linear head construction. converts LSTM output to desired shape

    # defines architecture of the model
    def forward(self, input):
        # input needs shape (batch_size, input_sequence_length, input_size)

        # input data passed into LSTM cell
        lstm_out, (h_n, c_n) = self.lstm(input.reshape(-1, self.input_sequence_length, self.input_size))

        # output data is result of final LSTM state passed through our linear head layer
        output = self.head(h_n[-1])

        return output