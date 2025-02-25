import torch
import torch.nn as nn


from scripts.kg_objects import Entity, Relation
from scripts.utils import DataFromJSON

class RotTransformer(nn.module):
    def __init__(self, model_conf: DataFromJSON, gpu_device: torch.device = None, dicts_set: dict = None):
        super(RotTransformer, self).__init__()
        ## Triplet Transformer Attributes
        self.triplet_trans_heads =  model_conf.triplet_trans_heads
        self.triplet_trans_layers = model_conf.triplet_trans_layers

        ## Graph Transformer Attributes

        ## Overall Model Settings
        self.n_embed = model_conf.n_embed
        self.epochs = model_conf.epochs
        self.gpu_device = gpu_device
        self.dicts_set = dicts_set

    def triplet_encoder(self, triplet: list[Entity]):
        triplet = [e.embed_attributes(self.attribute_weights, self.attribute_planes) for e in triplet if isinstance(e, Entity)]
        encoder_layer = nn.TransformerEncoderLayer(self.n_embed, self.triplet_trans_heads)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, self.triplet_trans_layers)

        input = [obj.embedding for obj in triplet]
        out = transformer_encoder(input) #! need check if triplet is correct size as an input

    def graph_transformer(self, entities: list[Entity], relations: list[Relation]):
        self.FF = nn.Linear(self.n_embed, self.n_embed)
    
    def self_attention_relation_bias(self, entities: list[Entity], relations: list[Relation]):
        pass

    def transformer_network(self, query, args):
        # query : ([Entity, Relation, Entity], int) where int is the location of the anchor

        contextual_triplets = self.context_subgraph(self, query[0][query[1]])
        anchor_entities = [ct[0][ct[1]] for ct in contextual_triplets]
        contextual_relations = [ct[0][1] for ct in contextual_triplets]

        for i, triple_information in enumerate(contextual_triplets):
            t = triple_information[0]
            t_enriched = self.triplet_encoder(t)
            anchor_entities.append(t_enriched[triple_information[1]])
            contextual_relations.append(t_enriched[1])

        out = self.graph_transformer(anchor_entities, contextual_relations)

    def context_subgraph(self, anchor: Entity) -> [([Entity, Relation, Entity], int)]:
        """
        Returns a list of tuples representing the contextual triplets given a anchor entity where the first
        item in the tuple is the triplet and the second item is the position of the anchor entity in the triplet

        """
        pass

class MultiHeadAttentionUniqueBias(nn.module):
    def __init__(self, num_heads, d_emb, d_k, d_v, relations):
        super(MultiHeadAttentionUniqueBias, self).__init__()

        self.W_Q = nn.Linear(d_emb, d_k)
        self.W_K = nn.linear(d_emb, d_k)
        self.W_V = nn.linear(d_k, d_v)
        self.W_O = nn.linear(d_v, d_emb)

        self.relations = relations

    def forward(self):
        pass
