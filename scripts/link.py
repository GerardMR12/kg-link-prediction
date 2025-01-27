import os
import math
import numpy as np
import torch
from time import sleep
from datetime import datetime

from scripts.graph import GraphInstance
from scripts.utils import DataFromJSON
from scripts.model import *

class LinkPrediction():
    """
    Link prediction class, which uses the knowledge graph and creates a model.
    """
    def __init__(self, graph: GraphInstance, conf: dict, debug: bool = False, gpu_device: torch.device = None, save_path: str = None):
        self.graph = graph
        self.debug = debug
        self.gpu_device = gpu_device
        self.conf = conf
        self.save_path = save_path
        self.model_conf = DataFromJSON(conf.models["name" == conf.using], "model_configuration")
        self.make_output_folder(self.save_path)
    
    def make_output_folder(self, path: str):
        """
        Make the output folder.
        """
        if not os.path.exists(path):
            os.makedirs(path)

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
        model = self.create_model(vocab_size).to(self.gpu_device)

        if not self.conf.training:
            # Load the model
            model.load_state_dict(torch.load(self.save_path + self.conf.using + "_trained.pth", weights_only=True))

            # Evaluate the model
            model = model.inference_model(links_states, link_to_int, int_to_link, vocab_size)
        else:
            # Train the model
            model = model.train_model(links_states, link_to_int, int_to_link, vocab_size)

            # Save model
            self.save_model(model)

    def create_model(self, vocab_size: int):
        """
        Create the link prediction model.
        """
        print("Creating the link prediction model...")

        # Model name
        if self.conf.using == "TripletSymmetricVAE":
            model = TripletSymmetricVAE(self.model_conf, vocab_size, self.gpu_device)
        else:
            raise ValueError(f"Model {self.model_name} is not implemented.")

        return model
    
    def save_model(self, model):
        """
        Save the model.
        """
        print("Saving the model...")

        # Date and time string
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%Hh-%Mm")

        # Save the model
        torch.save(model.state_dict(), self.save_path + self.conf.using + "_trained_" + dt_string + ".pth")
    
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