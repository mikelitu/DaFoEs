import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class LSTM(nn.Module):
    def __init__(self, hidden_layers: int = 64):
        super(LSTM, self).__init__()
        self.hidden_layers = hidden_layers
        self.lstm1 = nn.LSTMCell(1, self.hidden_layers)
        self.lstm2 = nn.LSTMCell(self.hidden_layers, self.hidden_layers)
        self.linear = nn.Linear(self.hidden_layers, 1)

    def forward(self, y: torch.Tensor, future_preds: int = 0) -> torch.Tensor:
        outputs, n_samples = [], y.size(0)
        h_t = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        c_t = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        h_t2 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)
        c_t2 = torch.zeros(n_samples, self.hidden_layers, dtype=torch.float32)

        for input_t in y.split(1, dim=1):
            #N, 1
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)
        
        for i in range(future_preds):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)
        
        outputs = torch.cat(outputs, dim=1)
        return outputs

def training_loop(n_epochs: int, model: LSTM, optimizer: torch.optim.LBFGS, loss_fn:torch.nn.MSELoss,
                    train_input: torch.Tensor, train_target: torch.Tensor, 
                    test_input: torch.Tensor, test_target: torch.Tensor):
    for i in range(n_epochs):
        def closure():
            optimizer.zero_grad()
            out = model(train_input)
            loss = loss_fn(out, train_target)
            loss.backward()
            return loss
        optimizer.step(closure)
        with torch.no_grad():
            future = 1000
            pred = model(test_input, future_preds=future)
            loss = loss_fn(pred[:, :-future], test_target)
            y = pred.detach().numpy()
        
        #draw figures
        plt.figure(figsize=(12,6))
        plt.title(f"Step {i+1}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        n = train_input.shape[1] #999
        def draw(yi, color):
            plt.plot(np.arange(n), yi[:n], color, linewidth=2.0)
            plt.plot(np.arange(n, n+future), color+":", linewidth=2.0)
        draw(y[0], 'r')
        draw(y[1], 'b')
        draw(y[2], 'g')
        plt.savefig("predict%d.png"%i, dpi=200)
        plt.close()
        out = model(train_input)
        loss_print = loss_fn(out, train_target)
        print("Step: {}, Loss: {}".format(i, loss_print))

def main():
    N = 100 #number of samples
    L = 1000 # length of each sampe (number of values each sine wave)
    T = 20 # width of the wave
    x = np.empty((N, L), np.float32) # instantiate empty array
    x[:] = np.arange((L)) + np.random.randint(-4*T, 4*T, N).reshape(N, 1)
    y = np.sin(x/1.0/T).astype(np.float32)
    train_input = torch.from_numpy(y[3:, :-1])
    train_target = torch.from_numpy(y[3:, 1:])
    test_input = torch.from_numpy(y[:3, :-1])
    test_target = torch.from_numpy(y[:3, 1:])
    model = LSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.08)
    training_loop(n_epochs=10, model=model, optimizer=optimizer, loss_fn=criterion,
    train_input=train_input, train_target=train_target, test_input=test_input, test_target=test_target)

if __name__ == "__main__":
    main()