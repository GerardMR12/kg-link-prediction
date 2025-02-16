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

# Define a function to retrieve credentials from environment variables
def get_credentials():
    if using_graph.lower() == "private":
        return {
            "uri": os.getenv("NEO4J_URI"),
            "username": os.getenv("NEO4J_USERNAME"),
            "password": os.getenv("NEO4J_PASSWORD")
        }
    elif using_graph.lower() == "neoflix":
        return {
            "uri": os.getenv("NEO4J_NEOFLIX_URI"),
            "username": os.getenv("NEO4J_NEOFLIX_USERNAME"),
            "password": os.getenv("NEO4J_NEOFLIX_PASSWORD")
        }
    elif using_graph.lower() == "movies":
        return {
            "uri": os.getenv("NEO4J_MOVIES_URI"),
            "username": os.getenv("NEO4J_MOVIES_USERNAME"),
            "password": os.getenv("NEO4J_MOVIES_PASSWORD")
        }
    else:
        raise ValueError("Unknown graph configuration")

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

        # Load environment variables from .env file
        load_dotenv()

        # Decide which graph to use based on an environment variable (or any configuration logic)
        using_graph = os.getenv("USING_GRAPH", "movies")  # default to private if not set

        # Get the credentials based on the chosen configuration
        credentials = get_credentials()

        # Optionally, avoid printing credentials to debug/log output
        debug = os.getenv("DEBUG", "False").lower() in ("true", "1")

        if debug:
            print("Using graph:", using_graph)
            print("Credentials loaded successfully (values hidden).")

        # Create a connection to the database
        driver = GraphDatabase.driver(credentials["uri"], auth=(credentials["username"], credentials["password"]))

        # Create a graph instance
        graph = GraphInstance(driver)

        # Create the link prediction object
        link = LinkPrediction(graph, conf, debug=debug, gpu_device=gpu_device, save_path=args.save)

        # Start the link prediction
        link.start()

    except Exception as e:
        traceback.print_exc()
        print("Error:", e)

    finally:
        driver.close()

