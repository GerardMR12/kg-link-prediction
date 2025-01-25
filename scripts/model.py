import torch
import torch.nn as nn

class LinkPredVAE(nn.Module):
    def __init__(self, n_embed: int, n_hidden: tuple[int], n_latent: int, vocab_size: int):
        super(LinkPredVAE, self).__init__()
        self.n_embed = n_embed
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.vocab_size = vocab_size

        self.build_model(n_embed, n_hidden, n_latent, vocab_size)

    def build_model(self, n_embed: int, n_hidden: tuple[int], n_latent: int, vocab_size: int):
        """
        Build the encoder, decoder, and edge decoder networks.
        """
        # Embedding layer
        self.embed = nn.Embedding(vocab_size, n_embed)

        # Encoder
        encoder_layers = []
        n_hidden_enc = [n_embed] + list(n_hidden)
        for i in range(len(n_hidden_enc) - 1):
            encoder_layers.append(nn.Linear(n_hidden_enc[i], n_hidden_enc[i + 1]))
            encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Linear(n_hidden_enc[-1], n_latent * 2))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        n_hidden_dec = [n_latent] + list(n_hidden)[::-1] + [n_embed]
        for i in range(len(n_hidden_dec) - 1):
            decoder_layers.append(nn.Linear(n_hidden_dec[i], n_hidden_dec[i + 1]))
            decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)

        # Edge decoder
        edge_hidden = int((n_embed + vocab_size) / 2)
        edge_hidden = max(edge_hidden, 2*n_embed)
        self.edge_decoder = nn.Sequential(
            nn.Linear(n_latent, edge_hidden),
            nn.Linear(edge_hidden, vocab_size)
        )

        # Softmax layer for edge probabilities
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, idx, noise_factor: float = 1.0):
        """
        Forward pass through the model.
        """
        x = self.embed(idx)
        mu, logvar = self.encoder(x).chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar, noise_factor)
        x_hat = self.decoder(z)
        edge_logits = self.edge_decoder(z)
        probs = self.softmax(edge_logits)
        return x, x_hat, edge_logits, probs, mu, logvar

    def reparameterize(self, mu, logvar, noise_factor: float = 1.0):
        """
        Reparameterization trick for sampling from a Gaussian.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) * noise_factor
        return mu + eps * std

    def loss(self, x, x_hat, edge_logits, mu, logvar):
        x_loss = nn.functional.binary_cross_entropy_with_logits(x_hat, x, reduction='sum')
        edge_loss = nn.functional.binary_cross_entropy_with_logits(edge_logits, x.new_zeros(x.shape), reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return x_loss + edge_loss + kl_div

    def generate(self, n_samples):
        """
        Generate samples from the model.
        """
        z = torch.randn(n_samples, self.n_latent)
        x = torch.sigmoid(self.decoder(z))
        return x

    def reconstruct(self, x):
        mu, _ = self.encoder(x).chunk(2, dim=1)
        z = self.reparameterize(mu, torch.zeros_like(mu))
        x_hat = torch.sigmoid(self.decoder(z))
        return x_hat