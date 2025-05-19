import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 50 -> 25

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 25 -> 12
        )

        self.rnn = nn.LSTM(
            input_size=128 * 12,
            hidden_size=nh,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(nh * 2, nclass)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        if h != 12:
            raise ValueError(f"Expected height=12 after conv, got {h}")
        conv = conv.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
        rnn_out, _ = self.rnn(conv)
        output = self.fc(rnn_out)
        return output.permute(1, 0, 2)  # (T, B, C)
