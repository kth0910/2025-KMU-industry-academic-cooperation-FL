import flwr as fl
import torch
import torch.nn.functional as F
from model import LSTMNextWordModel
from dataset import load_data

class FLClient(fl.client.NumPyClient):
    def __init__(self, cid, train_data, test_data, device):
        self.cid = cid
        self.train_data = train_data
        self.test_data = test_data
        # GPU 지원
        self.device = torch.device("cuda")
        self.model = LSTMNextWordModel().to(self.device)

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(self.device) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        lr = config.get("lr")
        epochs = config.get("epochs")
        batch_size = config.get("batch_size")
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.model.train()
        X, y = self.train_data
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits, _ = self.model(xb)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                optimizer.step()
        return self.get_parameters(), len(X), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        X, y = self.test_data
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=256)
        correct = total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits, _ = self.model(xb)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        return float(correct / total), total, {"accuracy": correct / total}


def client_fn(cid: str) -> FLClient:
    data = load_data()
    train_data = data["train"][cid]
    test_data = data["test"]
    return FLClient(cid, train_data, test_data, None)
