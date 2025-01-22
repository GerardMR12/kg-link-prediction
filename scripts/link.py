from scripts.graph import GraphInstance

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