import math
import numpy as np
import torch
from time import sleep

from scripts.graph import GraphInstance
from scripts.model import LinkPredVAE

class LinkPrediction():
    """
    Link prediction class, which uses the knowledge graph and creates a model.
    """
    def __init__(self, graph: GraphInstance, debug: bool = False):
        self.graph = graph
        self.debug = debug

    def start(self):
        """
        Start the link prediction.
        """
        print("Starting link prediction...")

        # Get the vocabulary sets
        links_states, link_to_int, int_to_link = self.get_links_vocab(self.graph)

        if self.debug:
            print("Lengths of links_states, link_to_int, int_to_link:", len(links_states), len(link_to_int), len(int_to_link))

        # Get the vocabulary size
        vocab_size = len(links_states)

        # Create the model
        model = self.create_model(vocab_size)

        # Train the model
        self.train_model(model, links_states, link_to_int, int_to_link, vocab_size)

    def create_model(self, vocab_size: int):
        """
        Create the link prediction model.
        """
        print("Creating the link prediction model...")

        # Define the model parameters
        n_embed = 256
        n_hidden = (256, 128, 256, 128)
        n_latent = 64

        # Create the model
        model = LinkPredVAE(n_embed, n_hidden, n_latent, vocab_size)

        return model

    def train_model(self, model: LinkPredVAE, links_states: dict, link_to_int: dict, int_to_link: dict, vocab_size: int):
        """
        Train the link prediction model.
        """
        print("Training the link prediction model...")

        # Define the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Define the loss function
        loss_fn = torch.nn.CrossEntropyLoss()
        loss_mse = torch.nn.MSELoss()

        # Define the number of epochs
        epochs = 10000

        # Get the links states with the value equals to 1
        value_1_links_states = {key: value for key, value in links_states.items() if value == 1}

        # Define the training loop
        for epoch in range(epochs):
            # Generate random samples
            input = torch.tensor([link_to_int[key] for key in value_1_links_states.keys()])

            # Randomly shuffle the indices
            input = input[torch.randperm(input.size(0))]

            # Forward pass
            x, x_hat, edge_logits, probs, mu, logvar = model(input)

            # Get the values which should be 1
            one_values = probs[np.arange(probs.shape[0]), input].unsqueeze(1)

            # Compute the loss
            embed_loss = loss_fn(x, x_hat)
            probs_loss = torch.abs(one_values - torch.ones_like(one_values)).sum()/len(one_values)
            loss = embed_loss + probs_loss

            # Print the loss
            print(f"Epoch: {epoch}, Total Loss: {loss.item():.5f}, Embedding Loss: {embed_loss.item():.5f}, Probs Loss: {probs_loss.item():.5f}")

            # Zero the gradients
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Optimize
            optimizer.step()
    
    def get_links_vocab(self, graph: GraphInstance):
        """
        Get the links vocabulary.
        """
        nodes = graph.get_nodes()
        nodes_dict = graph.get_nodes_dict()
        relationships = graph.get_relationships(distinct=True)
        relationships_dict = graph.get_relationships_dict(distinct=True)
        
        if self.debug:
            print("We have", len(nodes), "nodes and", len(relationships_dict), "distinc relationships in the graph.")
            print("This makes a vocabulary of", len(nodes)**2*len(relationships_dict), "elements.")

        # Find the links in the graph
        links = graph.get_links()

        # Create a dictionary of all possible relationships
        links_states = {}
        link_to_int = {}
        int_to_link = {}
        idx = 0
        for node1 in nodes:
            for node2 in nodes:
                for relationship in relationships:
                    links_states[(node1, relationship, node2)] = 1 if (node1, relationship, node2) in links else 0
                    link_to_int[(node1, relationship, node2)] = idx
                    int_to_link[idx] = (node1, relationship, node2)
                    idx += 1

        # Count the existing links
        existing_links = {key: value for key, value in links_states.items() if value == 1}

        if self.debug:
            print("We have", len(links_states), "possible relationships in the graph and", len(existing_links), " of them exist.")
            print("These are the existing links in the graph:", existing_links)

        return links_states, link_to_int, int_to_link