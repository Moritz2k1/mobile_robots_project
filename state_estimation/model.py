from torch import nn
import numpy as np
from matplotlib import pyplot as plt

class LSTM(nn.Module):

    def __init__(self, output_size, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.output_size = output_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        ula, (h_out, _) = self.lstm(x)

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)

        return out
    
    def train_loop(self, num_epochs, trainX, trainY, optimizer, criterion):
        delta_list = []
        for epoch in range(num_epochs):
            outputs = self.forward(trainX)
            optimizer.zero_grad()

            # obtain the loss function
            loss = criterion(outputs, trainY)

            loss.backward()

            optimizer.step()
            if epoch % 100 == 0:
                # delta = sum(abs(outputs - trainY)).item()
                # delta_list.append(delta)
                # print("Epoch: %d, loss: %1.5f, delta: %f" % (epoch, loss.item(), delta))
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
        # print("Median delta: %f" % (np.median(delta_list)))

    def test_loop(self, dataX, dataY, criterion):
        # set model to evaluation mode
        self.eval()

        train_predict = self.forward(dataX)
        mserror = criterion(train_predict, dataY)
        print("MSE for val data: %1.5f" % (mserror.item()))
        #print("Delta for val data: %f" % (sum(abs(train_predict - dataY))))

        return train_predict

        data_predict = train_predict.data.numpy()
        dataY_plot = dataY.data.numpy()

        # data_predict = sc.inverse_transform(data_predict)
        # dataY_plot = sc.inverse_transform(dataY_plot)

        plt.axvline(x=train_size, c='r', linestyle='--')

        plt.plot(dataY_plot)
        plt.plot(data_predict)
        plt.suptitle('Time-Series Prediction')
        plt.savefig('output_test.png')
        plt.show()