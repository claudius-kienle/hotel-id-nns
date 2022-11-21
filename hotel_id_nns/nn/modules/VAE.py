import copy
from typing import List, Tuple
import torch
from torch import nn
from hotel_id_nns.nn.modules.Decoder import Decoder

from hotel_id_nns.nn.modules.Encoder import Encoder


class VAE(nn.Module):

    def __init__(
        self,
        in_size: int,
        in_out_channels: int,
        hidden_channels: List[int],
        latent_dim: int,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(in_size=in_size,in_channels=in_out_channels, hidden_channels=hidden_channels)

        self.fc_mu = nn.Linear(
            in_features=hidden_channels[-1],
            out_features=latent_dim,
        )
        self.fc_logvar = nn.Linear(
            in_features=hidden_channels[-1],
            out_features=latent_dim,
        )

        self.decoder_input = nn.Linear(
            in_features=latent_dim,
            out_features=hidden_channels[-1],
        )

        rev_hidden_channels = copy.deepcopy(hidden_channels)
        rev_hidden_channels.reverse()

        self.decoder = Decoder(
            out_channels=in_out_channels,
            hidden_channels=rev_hidden_channels,
        )

    def _encode(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ls = self.encoder.forward(input)
        ls_flattened = ls.flatten(start_dim=1)

        mu = self.fc_mu(ls_flattened)
        logvar = self.fc_logvar(ls_flattened)

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(mu)
        return mu + noise * std

    def _decode(self, input: torch.Tensor) -> torch.Tensor:
        output = self.decoder_input(input)
        output = self.decoder(output)
        return output


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mu, logvar = self._encode(input)
        z = self.reparameterize(mu=mu, logvar=logvar)
        output = self._decode(z)
        return [output, mu, logvar]