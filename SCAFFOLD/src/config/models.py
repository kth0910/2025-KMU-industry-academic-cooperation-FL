from torch import nn

ARGS = {
    "mnist": (1, 256, 10),
    "emnist": (1, 256, 62),
    "fmnist": (1, 256, 10),
    "cifar10": (3, 400, 10),
    "cifar100": (3, 400, 100),
}


class LogisticRegression(nn.Module):
    def __init__(self, dataset: str):
        super().__init__()
        dataset_info = {
            "mnist": {"input_shape": [1, 28, 28], "num_classes": 10},
            "emnist": {"input_shape": [1, 28, 28], "num_classes": 62},
            "fmnist": {"input_shape": [1, 28, 28], "num_classes": 10},
            "cifar10": {"input_shape": [3, 32, 32], "num_classes": 10},
            "cifar100": {"input_shape": [3, 32, 32], "num_classes": 100},
        }

        if dataset not in dataset_info:
            raise ValueError(f"Unknown dataset: {dataset}")
        input_shape = dataset_info[dataset]["input_shape"]
        num_classes = dataset_info[dataset]["num_classes"]
        input_dim = 1
        for d in input_shape:
            input_dim *= d

        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 200),
            nn.ReLU(),
            nn.Linear(200, 62),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


class LeNet5(nn.Module):
    def __init__(self, dataset) -> None:
        super(LeNet5, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ARGS[dataset][0], 6, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(ARGS[dataset][1], 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, ARGS[dataset][2]),
        )

    def forward(self, x):
        return self.net(x)
