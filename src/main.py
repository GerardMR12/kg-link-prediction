import os
import torch
import traceback
from neo4j import GraphDatabase
from dotenv import load_dotenv
from scripts.graph import GraphInstance
from scripts.link import LinkPrediction

if __name__ == "__main__":
    try:
        # Load environment variables
        load_dotenv()

        # Define connection details
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")

        # Debug flag
        debug = False

        # Create a connection
        driver = GraphDatabase.driver(uri, auth=(username, password))

        # Check if CUDA is available
        print("CUDA available:", torch.cuda.is_available())

        # Check the GPU name
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))

        # Create a graph instance
        graph = GraphInstance(driver, build_graph=False)

        # Create the link prediction object
        link = LinkPrediction(graph, debug=debug)

        # Start the link prediction
        link.start()

    except Exception as e:
        traceback.print_exc()
        print("Error:", e)

    finally:
        driver.close()

