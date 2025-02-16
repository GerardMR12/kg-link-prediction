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
        
    def get_entities_property(self, property_name="id"):
        query = """
        MATCH (n)
        WHERE n is NOT NULL
        RETURN n
        """
        result = self.run_query(query)
        return [record["n"][property_name] for record in result]
    
    def get_entities(self, feature: str="elementId"):
        query = f"""
        MATCH (n)
        WHERE n is NOT NULL
        RETURN {feature}(n) AS id
        """
        return self.run_query(query)
    
    def get_entities_dict(self, feature: str="elementId"):
        entities = self.get_entities(feature=feature)
        return {f"{idx}": entity["id"] for idx, entity in enumerate(entities)}
    
    def get_links(self):
        query = """
        MATCH (n)-[r]->(m)
        WHERE n is NOT NULL AND m is NOT NULL
        RETURN elementId(n) AS id1, type(r) AS type, elementId(m) AS id2
        """
        result = self.run_query(query)
        return [(record["id1"], record["type"], record["id2"]) for record in result]
    
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