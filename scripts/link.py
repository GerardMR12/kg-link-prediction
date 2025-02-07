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

        # Create the model
        model = self.create_model().to(self.gpu_device)

        if not self.conf.training:
            # Load the model
            model.load_state_dict(torch.load(self.save_path + self.conf.using + "_trained.pth", weights_only=True))

            # Evaluate the model
            model = model.inference_model(noise_factor=self.model_conf.inference_noise)
        else:
            # Train the model
            model = model.train_model()

            # Save model
            self.save_model(model)

    def create_model(self):
        """
        Create the link prediction model.
        """
        print("Creating the link prediction model...")

        # Get the vocabulary sets
        dicts_set = self.get_dicts_set(self.graph, self.model_conf.needed_dicts)

        # Model name
        if self.conf.using == "TripletSymmetricVAE":
            model = TripletSymmetricVAE(self.model_conf, self.gpu_device, dicts_set)
        elif self.conf.using == "PartitionedSymmetricVAE":
            model = PartRotSymmetricVAE(self.model_conf, self.gpu_device, dicts_set)
        else:
            raise ValueError(f"Model {self.conf.using} is not implemented.")

        return model
    
    def save_model(self, model):
        """
        Save the model.
        """
        print("Saving the model...")

        # Date and time string
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%Hh-%Mm")

        # Ensure path exists
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Save the model
        torch.save(model.state_dict(), self.save_path + self.conf.using + "_trained_" + dt_string + ".pth")
    
    def get_dicts_set(self, graph: GraphInstance, selection: list = None) -> dict:
        """
        Get the links vocabulary.
        """
        dicts_set = {}

        # Basic features of the graph
        entities = graph.get_entities()
        entities_dict = graph.get_entities_dict()
        relationships = graph.get_relationships(distinct=True)
        relationships_dict = graph.get_relationships_dict(distinct=True)
        
        if "entity_to_int" in selection:
            dicts_set["entity_to_int"] = entities_dict
        if "relationship_to_int" in selection:
            relationships_dict = graph.get_relationships_dict(distinct=True)
            dicts_set["relationship_to_int"] = relationships_dict
        if "links_states" in selection:
            # Find the links in the graph
            links = graph.get_links()
            links_states = {}
            idx = 0
            for ent1 in entities:
                for ent2 in entities:
                    for relationship in relationships:
                        links_states[(ent1, relationship, ent2)] = 1 if (ent1, relationship, ent2) in links else 0
                        idx += 1
            dicts_set["links_states"] = links_states
        if "link_to_int" in selection:
            link_to_int = {}
            idx = 0
            for ent1 in entities:
                for ent2 in entities:
                    for relationship in relationships:
                        link_to_int[(ent1, relationship, ent2)] = idx
                        idx += 1
            dicts_set["link_to_int"] = link_to_int
        if "int_to_link" in selection:
            int_to_link = {}
            idx = 0
            for ent1 in entities:
                for ent2 in entities:
                    for relationship in relationships:
                        int_to_link[idx] = (ent1, relationship, ent2)
                        idx += 1
            dicts_set["int_to_link"] = int_to_link

        return dicts_set