import torch.nn as nn

# model wrapper
# linear regression fork for features and preset outputs


class regModel():
    def __init__(self, n_outcomes=1, covariance=None, weighted=False, features=None, model_name=None):
        self.n_outcomes = n_outcomes
        self.cov = covariance
        self.weighted = weighted
        self.features = ["NON-MEMBRANE"] if features is None else features

        print(f"MODEL\tInitializing  Regression model for {self.n_outcomes} outcome(s)")
        # features
        print(f"MODEL\tText features:\t{' + '.join(self.features)}")
        # longitude, latitude for n outcomes
        self.coord_output = self.n_outcomes * 2
        print(f"MODEL\tCoordinates:\t{self.coord_output}")
        # weights of gaussians
        self.weights_output = self.n_outcomes if self.weighted and self.n_outcomes > 1 else 0
        if self.weights_output > 0:
            print(f"MODEL\tWeights:\t{self.weights_output}")

        # covariance matrix
        self.covariances = {'spher': self.n_outcomes,
                            'diag': self.n_outcomes * 2,
                            'tied': 3,
                            'full': self.n_outcomes * 3}
        if self.cov is None:
            self.cov_output = 0
            print(f"MODEL\tNon-probabilistic model has been chosen")
        else:
            if self.cov not in self.covariances:
                self.cov = 'spher'
            self.cov_output = self.covariances[self.cov]
            print(f"MODEL\tCovariances:\t{self.cov_output}\tmatrix type:\t{self.cov}")

        self.original_model = model_name
        print(f"MODEL\tOriginal model to load:\t{self.original_model}")
        self.key_output = self.coord_output + self.weights_output + self.cov_output
        self.minor_output = 2
        self.minor_output += 1 if self.cov_output > 0 else 0

        self.feature_outputs = {}
        for f in range(len(self.features)):
            if f == 0:
                output = self.key_output
                print(f"MODEL\tKey feature \t{self.features[f]} outputs:\t{output}")
            else:
                output = self.minor_output
                print(f"MODEL\tMinor feature\t{self.features[f]} outputs:\t{output}")
            self.feature_outputs[self.features[f]] = output

        self.model = Regressor(self.original_model, self.feature_outputs)


class Regressor(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, output_size)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
            self.num_layers = num_layers

        def forward(self, outputs):
            for i in range(self.num_layers):
                outputs = self.fc1(outputs)
                outputs = self.relu(outputs)
                outputs = self.dropout(outputs)
            outputs = self.fc2(outputs)
            outputs = self.relu(outputs)
            outputs = self.fc3(outputs)
            return outputs
            self.key_regressor = nn.Linear(768, list(self.feature_outputs.values())[0])
            if len(self.feature_outputs) > 1:
                self.minor_regressor = nn.Linear(768, list(self.feature_outputs.values())[1])