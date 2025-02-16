import os
import json
import torch
import argparse
import torch
import traceback

from neo4j import GraphDatabase
from dotenv import load_dotenv
from scripts.utils import DataFromJSON
from scripts.graph import GraphInstance
from scripts.link import LinkPrediction

if __name__ == "__main__":
    try:
        # Check if CUDA is available
        print("CUDA available:", torch.cuda.is_available())

        # Check the GPU name
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))

        gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Parse the arguments
        parser = argparse.ArgumentParser(description="Link prediction model")
        parser.add_argument("--conf", help="Configuration file")
        parser.add_argument("--save", help="Saving file path")
        args = parser.parse_args()

        # Create configuration
        with open(args.conf, "r") as file:
            json_dict = json.load(file)
            conf = DataFromJSON(json_dict, "configuration_file")

        # Set the public credentials
        creden = {}
        creden["neoflix"] = {"uri": "neo4j+s://demo.neo4jlabs.com", "username": "neoflix", "password": "neoflix"}

        # Set the private credentials (given in .env file)
        load_dotenv()
        creden[".env"] = {"uri": os.getenv("NEO4J_URI"), "username": os.getenv("NEO4J_USERNAME"), "password": os.getenv("NEO4J_PASSWORD")}

        # Define connection details
        uri = os.getenv(creden[conf.using_graph]["uri"])
        username = os.getenv(creden[conf.using_graph]["username"])
        password = os.getenv(creden[conf.using_graph]["password"])

        # Debug flag
        debug = False

        # Create a connection
        driver = GraphDatabase.driver(uri, auth=(username, password))

        # Create a graph instance
        graph = GraphInstance(driver, build_graph=False)

        # Create the link prediction object
        link = LinkPrediction(graph, conf, debug=debug, gpu_device=gpu_device, save_path=args.save)

        # Start the link prediction
        link.start()

    except Exception as e:
        traceback.print_exc()
        print("Error:", e)

    finally:
        driver.close()

