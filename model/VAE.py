import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
from typing import List, Optional
from model.encoder import EncoderLSTM
from model.decoder import DecoderLSTM


class VAE(nn.Module):

    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, latent_dim=64, num_layers=2, dropout=0.3):
        
        super().__init__()

        self.encoder = EncoderLSTM(vocab_size, embed_dim, hidden_dim, latent_dim, num_layers, dropout)
        self.decoder = DecoderLSTM(vocab_size, embed_dim, hidden_dim, latent_dim, num_layers, dropout)
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size

    def reparametrize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std
    
    def forward(self, src, tgt):

        mu, logvar = self.encoder(src)
        z = self.reparametrize(mu, logvar)

        logits, _, _ = self.decoder(tgt, z)

        return logits, mu, logvar
    
    @torch.jit.export
    def generate(
        self,
        start_token: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        max_len: int = 512,
        temperature: float = 1.0,
    ) -> torch.Tensor:

        # Accept both (batch,) and (batch, 1) start tokens for C++ convenience.
        if start_token.dim() == 2 and start_token.size(1) == 1:
            start_token = start_token.squeeze(1)

        batch_size = start_token.size(0)
        device = start_token.device

        if z is None:
            z = torch.randn((batch_size, self.latent_dim), device=device)

        hidden = torch.jit.annotate(Optional[torch.Tensor], None)
        cell = torch.jit.annotate(Optional[torch.Tensor], None)
        x = start_token.unsqueeze(1)  # (batch, 1)

        generated = torch.jit.annotate(List[torch.Tensor], [])
        generated.append(start_token)

        for _ in range(max_len):
            logits, hidden, cell = self.decoder(x, z, hidden, cell)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

            generated.append(next_token)
            x = next_token.unsqueeze(1)

        return torch.stack(generated, dim=1)
    
def vae_loss(logits, targets, mu, logvar, pad_id, beta=1.0):

    reconstruction_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_id,
        reduction='sum') / targets.size(0)  

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / targets.size(0)

    return reconstruction_loss + beta * kl_loss, reconstruction_loss, kl_loss 

def attribute_regularization_loss(latent_vectors, attribute_values, reg_dim, delta=10.0):

    if attribute_values.dim() != 1:
        raise ValueError("attribute_values must be 1D: (batch,)")
    if reg_dim < 0 or reg_dim >= latent_vectors.size(1):
        raise ValueError(f"reg_dim={reg_dim} out of range for latent dim={latent_vectors.size(1)}")

    da = attribute_values.unsqueeze(1) - attribute_values.unsqueeze(0)
    dr = latent_vectors[:, reg_dim].unsqueeze(1) - latent_vectors[:, reg_dim].unsqueeze(0)

    return torch.mean(torch.abs(torch.tanh(delta * dr) - torch.sign(da)))


def ar_vae_loss(
    logits,
    targets,
    mu,
    logvar,
    pad_id,
    attributes,
    reg_dims,
    beta=1.0,
    gamma=1.0,
    delta=10.0,
):

    vae_total, reconstruction_loss, kl_loss = vae_loss(logits, targets, mu, logvar, pad_id, beta=beta)

    if attributes.dim() != 2:
        raise ValueError("attributes must be 2D: (batch, num_attributes)")
    if len(reg_dims) != attributes.size(1):
        raise ValueError("len(reg_dims) must match number of attribute columns")

    reg_loss = torch.tensor(0.0, device=mu.device)
    for attr_idx, reg_dim in enumerate(reg_dims):
        reg_loss = reg_loss + attribute_regularization_loss(
            latent_vectors=mu,
            attribute_values=attributes[:, attr_idx],
            reg_dim=reg_dim,
            delta=delta,
        )

    reg_loss = reg_loss / max(1, len(reg_dims))
    total_loss = vae_total + gamma * reg_loss

    return total_loss, reconstruction_loss, kl_loss, reg_loss
    
