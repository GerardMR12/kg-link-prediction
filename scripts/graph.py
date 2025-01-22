class GraphInstance():
    """
    Class to deal with Neo4j Aura instance and knowledge graph.
    """
    def __init__(self, driver, build_graph=False):
        self.driver = driver
        if build_graph:
            self.build_graph()

    def run_query(self, query, **kwargs):
        with self.driver.session() as session:
            result = session.run(query, **kwargs)
            return result.data()
        
    def build_graph(self):
        # Delete all nodes and relationships
        cypher_query = """
        MATCH (n)
        DETACH DELETE n
        """
        self.run_query(cypher_query)

        creation_query = """
        CREATE (eDogs:Animal {name: "Dogs"})
        CREATE (eCats:Animal {name: "Cats"})
        CREATE (eMice:Animal {name: "Mice"})
        CREATE (eHorses:Animal {name: "Horses"})
        CREATE (eBirds:Animal {name: "Birds"})
        CREATE (eFish:Animal {name: "Fish"})
        CREATE (eGrass:Plants {name: "Grass"})
        CREATE (eFruits:Plants {name: "Fruits"})
        CREATE (eVegetables:Plants {name: "Vegetables"})
        CREATE (eTrees:Plants {name: "Trees"})
        CREATE (eFlowers:Plants {name: "Flowers"})
        CREATE (eBees:Insects {name: "Bees"})
        CREATE (eAnts:Insects {name: "Ants"})
        CREATE (eButterflies:Insects {name: "Butterflies"})
        CREATE (eJungle:Zone {name: "Jungle"})
        CREATE (eDesert:Zone {name: "Desert"})
        CREATE (eOcean:Zone {name: "Ocean"})
        CREATE (eSavannah:Zone {name: "Savannah"})
        CREATE (eForest:Zone {name: "Forest"})
        CREATE (eDomestic:Animal_kind {name: "Domestic"})
        CREATE (eWild:Animal_kind {name: "Wild"})
        CREATE (eTall:Plants_Height {name: "Tall"})
        CREATE (eShort:Plants_Height {name: "Short"})
        CREATE (eSmall:Animal_Size {name: "Small"})
        CREATE (eMedium:Animal_Size {name: "Medium"})
        CREATE (eLarge:Animal_Size {name: "Large"})
        """
        self.run_query(creation_query)

        info_text = """
        MATCH (eDogs:Animal {name: "Dogs"}), (eCats:Animal {name: "Cats"})
        MERGE (eDogs)-[:EAT]->(eCats);

        MATCH (eCats:Animal {name: "Cats"}), (eMice:Animal {name: "Mice"})
        MERGE (eCats)-[:EAT]->(eMice);

        MATCH (eBirds:Animal {name: "Birds"}), (eAnts:Insects {name: "Ants"})
        MERGE (eBirds)-[:EAT]->(eAnts);

        MATCH (eMice:Animal {name: "Mice"}), (eSavannah:Zone {name: "Savannah"})
        MERGE (eMice)-[:LIVE_IN]->(eSavannah);

        MATCH (eHorses:Animal {name: "Horses"}), (eGrass:Plants {name: "Grass"})
        MERGE (eHorses)-[:EAT]->(eGrass);

        MATCH (eFish:Animal {name: "Fish"}), (eOcean:Zone {name: "Ocean"})
        MERGE (eFish)-[:LIVE_IN]->(eOcean);

        MATCH (eTrees:Plants {name: "Trees"}), (eForest:Zone {name: "Forest"})
        MERGE (eTrees)-[:FOUND_IN]->(eForest);

        MATCH (eFlowers:Plants {name: "Flowers"}), (eBees:Insects {name: "Bees"})
        MERGE (eFlowers)-[:POLLINATED_BY]->(eBees);

        MATCH (eFruits:Plants {name: "Fruits"}), (eForest:Zone {name: "Forest"})
        MERGE (eFruits)-[:FOUND_IN]->(eForest);

        MATCH (eButterflies:Insects {name: "Butterflies"}), (eFlowers:Plants {name: "Flowers"})
        MERGE (eButterflies)-[:VISIT]->(eFlowers);

        MATCH (eBees:Insects {name: "Bees"}), (eJungle:Zone {name: "Jungle"})
        MERGE (eBees)-[:LIVE_IN]->(eJungle);

        MATCH (eAnts:Insects {name: "Ants"}), (eDesert:Zone {name: "Desert"})
        MERGE (eAnts)-[:LIVE_IN]->(eDesert);

        MATCH (eDogs:Animal {name: "Dogs"}), (eDomestic:Animal_kind {name: "Domestic"})
        MERGE (eDogs)-[:CLASSIFIED_AS]->(eDomestic);

        MATCH (eCats:Animal {name: "Cats"}), (eDomestic:Animal_kind {name: "Domestic"})
        MERGE (eCats)-[:CLASSIFIED_AS]->(eDomestic);

        MATCH (eHorses:Animal {name: "Horses"}), (eDomestic:Animal_kind {name: "Domestic"})
        MERGE (eHorses)-[:CLASSIFIED_AS]->(eDomestic);

        MATCH (eMice:Animal {name: "Mice"}), (eWild:Animal_kind {name: "Wild"})
        MERGE (eMice)-[:CLASSIFIED_AS]->(eWild);

        MATCH (eBirds:Animal {name: "Birds"}), (eWild:Animal_kind {name: "Wild"})
        MERGE (eBirds)-[:CLASSIFIED_AS]->(eWild);

        MATCH (eGrass:Plants {name: "Grass"}), (eShort:Plants_Height {name: "Short"})
        MERGE (eGrass)-[:CLASSIFIED_AS]->(eShort);

        MATCH (eTrees:Plants {name: "Trees"}), (eTall:Plants_Height {name: "Tall"})
        MERGE (eTrees)-[:CLASSIFIED_AS]->(eTall);

        MATCH (eFlowers:Plants {name: "Flowers"}), (eShort:Plants_Height {name: "Short"})
        MERGE (eFlowers)-[:CLASSIFIED_AS]->(eShort);

        MATCH (eFish:Animal {name: "Fish"}), (eSmall:Animal_Size {name: "Small"})
        MERGE (eFish)-[:CLASSIFIED_AS]->(eSmall);

        MATCH (eDogs:Animal {name: "Dogs"}), (eMedium:Animal_Size {name: "Medium"})
        MERGE (eDogs)-[:CLASSIFIED_AS]->(eMedium);

        MATCH (eHorses:Animal {name: "Horses"}), (eLarge:Animal_Size {name: "Large"})
        MERGE (eHorses)-[:CLASSIFIED_AS]->(eLarge);

        MATCH (eBirds:Animal {name: "Birds"}), (eForest:Zone {name: "Forest"})
        MERGE (eBirds)-[:FOUND_IN]->(eForest);

        MATCH (eAnts:Insects {name: "Ants"}), (eFruits:Plants {name: "Fruits"})
        MERGE (eAnts)-[:VISIT]->(eFruits);

        MATCH (eButterflies:Insects {name: "Butterflies"}), (eFlowers:Plants {name: "Flowers"})
        MERGE (eButterflies)-[:POLLINATED_BY]->(eFlowers);

        MATCH (eJungle:Zone {name: "Jungle"}), (eWild:Animal_kind {name: "Wild"})
        MERGE (eJungle)-[:CONTAINS]->(eWild);

        MATCH (eSavannah:Zone {name: "Savannah"}), (eGrass:Plants {name: "Grass"})
        MERGE (eSavannah)-[:CONTAINS]->(eGrass);

        MATCH (eOcean:Zone {name: "Ocean"}), (eFish:Animal {name: "Fish"})
        MERGE (eOcean)-[:CONTAINS]->(eFish);

        MATCH (eForest:Zone {name: "Forest"}), (eTrees:Plants {name: "Trees"})
        MERGE (eForest)-[:CONTAINS]->(eTrees)
        """

        merging_queries = info_text.split(";")
        for query in merging_queries:
            self.run_query(query)

    def get_nodes(self):
        query = """
        MATCH (n)
        WHERE n is NOT NULL
        RETURN n
        """
        result = self.run_query(query)
        return [record["n"]["name"] for record in result]
    
    def get_nodes_dict(self):
        nodes = self.get_nodes()
        return {node: idx for idx, node in enumerate(nodes)}
    
    def get_links(self):
        query = """
        MATCH (n)-[r]->(m)
        WHERE n is NOT NULL AND m is NOT NULL
        RETURN n, type(r), m
        """
        result = self.run_query(query)
        return [(record["n"]["name"], record["type(r)"], record["m"]["name"]) for record in result]
    
    def get_relationships(self, distinct=False):
        distinct_text = "DISTINCT" if distinct else ""
        query = f"""
        MATCH (n)-[r]->(m)
        WHERE n is NOT NULL AND m is NOT NULL
        RETURN {distinct_text} type(r)
        """
        result = self.run_query(query)
        return [record["type(r)"] for record in result]
    
    def get_relationships_dict(self, distinct=False):
        relationships = self.get_relationships(distinct=distinct)
        return {relationship: idx for idx, relationship in enumerate(relationships)}