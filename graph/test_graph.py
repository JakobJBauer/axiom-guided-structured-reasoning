# We buld the graph from the master thesis proposal example codebook

from graph import Node, Edge, Graph
from graph_metrics import GraphMetrics

LEAF_NODES = ["short", "noun", "magical", "serious"]
ALL_NODES = LEAF_NODES + ["non-noun", "dense", "thrilling", "engaging"]

nodes = [
    Node("short"),
    Node("noun"),
    Node("magical"),
    Node("serious"),
    Node("non-noun", formula=lambda x: not x["noun"]),
    Node("dense", formula=lambda x: x["short"] and x["non-noun"]),
    Node("thrilling", formula=lambda x: x["magical"] or x["serious"]),
    Node("engaging", formula=lambda x: x["dense"] and x["thrilling"])
]

edges = [
    Edge("short", "dense"),
    Edge("noun", "non-noun"),
    Edge("non-noun", "dense"),
    Edge("dense", "engaging"),
    Edge("magical", "thrilling"),
    Edge("serious", "thrilling"),
    Edge("thrilling", "engaging")
]

codebook1_graph = Graph(nodes, edges)

TESTSET = [
    # short, noun, magical, serious | non-noun, dense, thrilling, engaging -> In accordance to ID
    (True, True, True, True, False, False, True, False),
    (True, True, True, False, False, False, True, False),
    (True, True, False, True, False, False, True, False),
    (True, True, False, False, False, False, True, False),
    (True, False, True, True, True, True, True, True),
    (True, False, True, False, True, True, True, True),
    (True, False, False, True, True, True, False, False),
    (True, False, False, False, True, True, False, False),
    (False, True, True, True, False, False, True, False),
    (False, True, True, False, False, False, True, False),
    (False, True, False, True, False, False, True, False),
    (False, True, False, False, False, False, True, False),
    (False, False, True, True, True, False, True, False),
    (False, False, True, False, True, False, True, False),
    (False, False, False, True, True, False, False, False),
    (False, False, False, False, True, False, False, False)
]

NUM_TESTS = 1
for testnum, test in enumerate(TESTSET[:NUM_TESTS]):
    for i, node in enumerate(LEAF_NODES): codebook1_graph.get_node_by_id(node).set_value(test[i]) # Set leaf values as in test
    codebook1_graph.auto_infer_values()
    print(f"codebook1_graph: {codebook1_graph}")

    reference_graph = codebook1_graph.copy()
    for i, node in enumerate(ALL_NODES): reference_graph.get_node_by_id(node).set_value(True)
    print(f"reference_graph: {reference_graph}")

    metrics = GraphMetrics(reference_graph, codebook1_graph)
    print(f"metrics only structure: \n{metrics.print_all_metrics()}\n")
    print(f"metrics with value checking: \n{metrics.print_all_metrics(check_values=True)}")


    