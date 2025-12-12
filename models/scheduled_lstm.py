import random

import torch
import torch.nn as nn

from models.simple_lstm import SimpleLSTM


class ScheduledSamplingLSTM(SimpleLSTM):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 50,
        num_layers: int = 2,
        output_size: int = 1,
    ):
        super().__init__(input_size, hidden_size, num_layers, output_size)
        self.lstm_cells = nn.ModuleList(
            [nn.LSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, teacher_forcing_ratio: float = 1.0) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        h = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size).to(x.device) for _ in range(self.num_layers)]

        outputs: list[torch.Tensor] = []

        current_input = x[:, 0, :]

        for t in range(seq_len):
            if t > 0:
                use_teacher_forcing = random.random() < teacher_forcing_ratio
                if use_teacher_forcing:
                    current_input = x[:, t, :]
                else:
                    current_input = outputs[-1]

            layer_input = current_input
            for i, cell in enumerate(self.lstm_cells):
                h[i], c[i] = cell(layer_input, (h[i], c[i]))
                layer_input = h[i]

            prediction = self.fc(h[-1])
            outputs.append(prediction)

        return torch.stack(outputs, dim=1)
