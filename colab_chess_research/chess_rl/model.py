"""Joint policy/value residual network; no training libraries required to import."""

from torch import nn

DEFAULT_ARCHITECTURE = dict(
    residual_blocks=6,
    channels=128,
    value_head_hidden=256,
    activation="relu",
    use_batch_norm=True,
    dropout=0.0,
)


def activation(name):
    return {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[name]()


def normalization(channels, enabled):
    return nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1) if enabled else nn.Identity()


class ResidualBlock(nn.Module):
    def __init__(self, channels, act, bn, dropout):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=not bn),
            normalization(channels, bn),
            activation(act),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1, bias=not bn),
            normalization(channels, bn),
        )
        self.activation = activation(act)

    def forward(self, x):
        return self.activation(x + self.branch(x))


class ChessPolicyValueNetSmall(nn.Module):
    def __init__(
        self,
        residual_blocks=6,
        channels=128,
        value_head_hidden=256,
        activation="relu",
        use_batch_norm=True,
        dropout=0.0,
    ):
        super().__init__()
        self.architecture = dict(
            residual_blocks=residual_blocks,
            channels=channels,
            value_head_hidden=value_head_hidden,
            activation=activation,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
        )
        act = globals()["activation"]
        self.stem = nn.Sequential(
            nn.Conv2d(21, channels, 3, padding=1, bias=not use_batch_norm),
            normalization(channels, use_batch_norm),
            act(activation),
        )
        self.trunk = nn.Sequential(
            *(
                ResidualBlock(channels, activation, use_batch_norm, dropout)
                for _ in range(residual_blocks)
            )
        )
        self.policy_head = nn.Conv2d(channels, 73, 1)
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=not use_batch_norm),
            normalization(32, use_batch_norm),
            act(activation),
            nn.Flatten(),
            nn.Linear(2048, value_head_hidden),
            act(activation),
            nn.Linear(value_head_hidden, 1),
            nn.Tanh(),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)

    def forward(self, x):
        trunk = self.trunk(self.stem(x))
        return self.policy_head(trunk).flatten(1), self.value_head(trunk).squeeze(-1)
