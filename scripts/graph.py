class GraphInstance():
    """
    Class to deal with Neo4j Aura instance and knowledge graph.
    """
    def __init__(self, driver):
        self.driver = driver

    def run_query(self, query, **kwargs):
        with self.driver.session() as session:
            result = session.run(query, **kwargs)
            return result.data()
        
    def get_entities(self):
        query = """
        MATCH (n)
        WHERE n is NOT NULL
        RETURN n
        """
        result = self.run_query(query)
        print(result)
        return [record["n"]["name"] for record in result]
    
    def get_entities_dict(self):
        entities = self.get_entities()
        return {entity: idx for idx, entity in enumerate(entities)}
    
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