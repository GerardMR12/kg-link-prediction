import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
from scripts.graph import GraphInstance
from scripts.link import LinkPrediction

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Define connection details
    uri = "neo4j+s://" + os.getenv("NEO4J_AURA_INSTANCE") + ".databases.neo4j.io"
    username = os.getenv("NEO4J_AURA_USERNAME")
    password = os.getenv("NEO4J_AURA_PASSWORD")

    # Debug flag
    debug = False

    # Create a connection
    driver = GraphDatabase.driver(uri, auth=(username, password))

    # Create a graph instance
    graph = GraphInstance(driver, build_graph=False)

    # Create the link prediction object
    link = LinkPrediction(graph, debug=debug)

    # Start the link prediction
    link.start()

