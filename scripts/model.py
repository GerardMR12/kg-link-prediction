import torch
import torch.nn as nn

from scripts.utils import DataFromJSON

class TripletSymmetricVAE(nn.Module):
    def __init__(self, model_conf: DataFromJSON, vocab_size: int, gpu_device: torch.device = None):
        super(TripletSymmetricVAE, self).__init__()
        self.n_embed = model_conf.n_embed
        self.n_hidden = model_conf.n_hidden
        self.n_latent = model_conf.n_latent
        self.epochs = model_conf.epochs
        self.gpu_device = gpu_device
        self.vocab_size = vocab_size

        self.build_model(self.n_embed, self.n_hidden, self.n_latent, vocab_size)

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
    
    def train_model(self, links_states: dict, link_to_int: dict, int_to_link: dict, vocab_size: int):
        """
        Train the link prediction model.
        """
        print("Training the link prediction model...")

        # Set the model to training mode
        self.train()

        # Define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        # Define the loss function
        loss_cre = torch.nn.CrossEntropyLoss().to(self.gpu_device)
        loss_mse = torch.nn.MSELoss().to(self.gpu_device)

        # Get the links states with the value equals to 1
        value_1_links_states = {key: value for key, value in links_states.items() if value == 1}

        # Define the training loop
        for epoch in range(self.epochs):
            # Generate random samples
            input = torch.tensor([link_to_int[key] for key in value_1_links_states.keys()])

            # Randomly shuffle the indices
            input = input[torch.randperm(input.size(0))].to(self.gpu_device)

            # Forward pass
            x, x_hat, edge_logits, probs, mu, logvar = self(input)

            # Get the values which should be 1
            iden = torch.eye(vocab_size).to(self.gpu_device)
            exp_probs = iden[input, :]

            # Compute the loss
            embed_loss = loss_mse(x, x_hat) # mse or cre?
            probs_loss = loss_cre(edge_logits, exp_probs)
            loss = embed_loss + probs_loss

            # Print the loss
            print(f"Epoch: {epoch}, Total Loss: {loss.item():.5f}, Embedding Loss: {embed_loss.item():.5f}, Probs Loss: {probs_loss.item():.5f}")

            # Zero the gradients
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Optimize
            optimizer.step()

        return self
    
    def inference_model(self, links_states: dict, link_to_int: dict, int_to_link: dict, vocab_size: int):
        """
        Evaluate the link prediction model.
        """
        print("Evaluating the link prediction model...")

        # Set the model to evaluation mode
        self.eval()

        # Get the links states with the value equals to 1
        value_1_links_states = {key: value for key, value in links_states.items() if value == 1}

        # Obtain inputs
        input = torch.tensor([link_to_int[key] for key in value_1_links_states.keys()])

        # # Print the following links: [1129, 3856, 756]
        # print(int_to_link[1129], int_to_link[3856], int_to_link[756])
        # print(links_states[int_to_link[1129]], links_states[int_to_link[3856]], links_states[int_to_link[756]])

        for i in range(input.size(0)):
            # Forward pass
            x, x_hat, edge_logits, probs, mu, logvar = self(input[i], noise_factor=1000.0)

            # Obtain top 3 predictions
            top_3 = torch.topk(probs, 3)

            # Print the top 3 predictions values and indices
            print("Top 3 predictions values:", [top_3.values[i].item() for i in range(top_3.values.size(0))])
            print("Top 3 predictions indices:", [top_3.indices[i].item() for i in range(top_3.indices.size(0))])

        return self