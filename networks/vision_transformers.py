from torch import nn, Tensor


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_dim: int | None = None,
        layer_norm_eps: float = 1e-5,
        dropout: float = 0.0,
        pretrained_id: str | None = None,
    ) -> None:
        super().__init__()
        if pretrained_id is not None:
            # TODO: load the pretrained weights and ignore all other arguments
            self.model = None
            return

        self.model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=mlp_dim if mlp_dim is not None else 2 * d_model,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.init_weights()

    def init_weights(self) -> None:
        for p in self.encoder.parameters():
            if p.dim() > 1:  # skip biases
                nn.init.xavier_uniform_(p)

    def forward(self, x: Tensor) -> Tensor:
        """Processes a set of encoded patches.

        Args:
            x (Tensor): `(bs, num_patches, d_model)` Input tokens.

        Returns:
            Tensor: `(bs, num_patches, d_model)` Token representations.
        """
        return self.model(x)
