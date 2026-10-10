import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_layers: list):
        super(Generator, self).__init__()
        self._hidden_layers = hidden_layers
        layers = []
        in_features = in_features + 1

        for i, (in_feature, out_feature) in enumerate(
                zip([in_features] + self._hidden_layers, self._hidden_layers + [out_features])):
            layers.append(nn.Linear(in_feature, out_feature))
            if i < len(self._hidden_layers):
                layers.append(nn.ReLU())
            else:
                layers.append(nn.LeakyReLU(0.2))

        self.layers = nn.Sequential(*layers)

    def forward(self, z, label) -> torch.Tensor:
        return self.layers(torch.cat((z, label), -1))


class CDiscriminator(nn.Module):
    def __init__(self, input_features: int, hidden_layers: list):
        super(CDiscriminator, self).__init__()
        self.input_features = input_features + 1
        self.hidden_layers = hidden_layers

        layers = []

        for i in range(len(self.hidden_layers)):
            if i == 0:
                layers.append(nn.Linear(in_features=self.input_features, out_features=self.hidden_layers[i]))
            else:
                layers.append(nn.Linear(in_features=self.hidden_layers[i - 1], out_features=self.hidden_layers[i]))
            layers.append(nn.LeakyReLU(0.2))

        self.layers = nn.Sequential(*layers)
        self.fc_source = nn.Linear(self.hidden_layers[-1], 1)

    def forward(self, data: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.fc_source(self.layers(torch.cat((data, labels), -1)))

class Auxiliary_Classifier(nn.Module):
    def __init__(self, num_classes: int, input_features: int, hidden_layers: list):
        super(Auxiliary_Classifier, self).__init__()
        self.input_features = input_features
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers

        self.cls_sigmoid = nn.Sigmoid()
        layers = []

        for i in range(len(self.hidden_layers)):
            if i == 0:
                layers.append(nn.Linear(in_features=self.input_features, out_features=self.hidden_layers[i]))
            else:
                layers.append(nn.Linear(in_features=self.hidden_layers[i - 1], out_features=self.hidden_layers[i]))
            layers.append(nn.LeakyReLU(0.2))

        self.layers = nn.Sequential(*layers)
        self.fc_class = nn.Linear(self.hidden_layers[-1], num_classes - 1)

    def forward(self, data: torch.Tensor) -> torch.Tensor:

        return self.cls_sigmoid(self.fc_class(self.layers(data)))

class Auxiliary_Classifier2(nn.Module):
    def __init__(self, num_classes: int, input_features: int, hidden_layers: list):
        super(Auxiliary_Classifier2, self).__init__()
        self.input_features = input_features
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers

        self.cls_sigmoid = nn.Sigmoid()
        layers = []

        for i in range(len(self.hidden_layers)):
            if i == 0:
                layers.append(nn.Linear(in_features=self.input_features, out_features=self.hidden_layers[i]))
            else:
                layers.append(nn.Linear(in_features=self.hidden_layers[i - 1], out_features=self.hidden_layers[i]))
            layers.append(nn.LeakyReLU(0.2))

        self.layers = nn.Sequential(*layers)
        self.fc_class = nn.Linear(self.hidden_layers[-1], num_classes - 1)
  

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.cls_sigmoid(self.fc_class(self.layers(data)))



