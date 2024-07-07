import torch
import torch.nn as nn

class Agent(nn.Module):
    input_size = 2
    hidden_size = 2
    output_size = 1
    num_layers = 1

    def __init__(self, rnn_weights_ih, rnn_weights_hh, rnn_bias_ih, rnn_bias_hh, output_weights, output_bias):
        super(Agent, self).__init__()
        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        # Initialize weights and biases
        self.rnn.weight_ih_l0.data = rnn_weights_ih
        self.rnn.weight_hh_l0.data = rnn_weights_hh
        self.rnn.bias_ih_l0.data = rnn_bias_ih
        self.rnn.bias_hh_l0.data = rnn_bias_hh
        self.out.weight.data = output_weights
        self.out.bias.data = output_bias

        # freeze parameters
        for param in self.rnn.parameters():
            param.requires_grad = False
        for param in self.out.parameters():
            param.requires_grad = False

        # flatten parameters
        self.rnn.flatten_parameters()
    
    @torch.no_grad()
    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        output = self.out(output)
        output = torch.sigmoid(output).round() * 2 - 1 # convert to -1 or 1
        return output, hidden

class AgentNoMemory(nn.Module):
    input_size = 2
    output_size = 1

    def __init__(self, output_weights, output_bias):
        super(AgentNoMemory, self).__init__()
        self.out = nn.Linear(self.input_size, self.output_size)

        # Initialize weights and biases
        self.out.weight.data = output_weights
        self.out.bias.data = output_bias

        # freeze parameters
        for param in self.out.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def forward(self, input, hidden=None):
        output = self.out(input)
        output = torch.sigmoid(output).round() * 2 - 1 # convert to -1 or 1
        return output, hidden

class SimpleAgent(nn.Module):
    """
    A simple agent that only uses the opponent's action as input.
    """
    input_size = 1
    output_size = 1

    def __init__(self, output_weights, output_bias):
        super(SimpleAgent, self).__init__()
        self.out = nn.Linear(self.input_size, self.output_size)

        # Initialize weights and biases
        self.out.weight.data = output_weights
        self.out.bias.data = output_bias

        # freeze parameters
        for param in self.out.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, input, hidden=None):
        input = input[:, 1].unsqueeze(1) # only use opponent's action
        output = self.out(input)
        # output = torch.sigmoid(output).round() * 2 - 1 # convert to -1 or 1
        output = (output>=0).int() * 2 -1  # convert to -1 or 1
        return output, hidden