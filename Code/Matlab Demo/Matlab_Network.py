import matplotlib.pyplot as plt
import networkx as nx


def create_network_diagram():
    # Create a directed graph
    G = nx.DiGraph()

    # Define the layers
    layers = [
        ("Input", "[imgSize(1) x imgSize(2) x 1]"),
        ("Conv2D_1", "Conv2D (3x3, 32 filters)"),
        ("ReLU_1", "ReLU"),
        ("Conv2D_2", "Conv2D (3x3, 32 filters)"),
        ("BatchNorm_1", "BatchNorm"),
        ("ReLU_2", "ReLU"),
        ("MaxPooling_1", "MaxPooling (2x2, stride=2)\nOutput: 64x64x32"),

        ("Conv2D_3", "Conv2D (3x3, 64 filters)"),
        ("ReLU_3", "ReLU"),
        ("Conv2D_4", "Conv2D (3x3, 64 filters)"),
        ("BatchNorm_2", "BatchNorm"),
        ("ReLU_4", "ReLU"),
        ("MaxPooling_2", "MaxPooling (2x2, stride=2)\nOutput: 32x32x64"),

        ("Conv2D_5", "Conv2D (3x3, 128 filters)"),
        ("ReLU_5", "ReLU"),
        ("Conv2D_6", "Conv2D (3x3, 128 filters)"),
        ("BatchNorm_3", "BatchNorm"),
        ("ReLU_6", "ReLU"),
        ("MaxPooling_3", "MaxPooling (2x2, stride=2)\nOutput: 16x16x128"),

        ("Conv2D_7", "Conv2D (3x3, 256 filters)"),
        ("ReLU_7", "ReLU"),
        ("Conv2D_8", "Conv2D (3x3, 256 filters)"),
        ("BatchNorm_4", "BatchNorm"),
        ("ReLU_8", "ReLU"),
        ("MaxPooling_4", "MaxPooling (2x2, stride=2)\nOutput: 8x8x256"),

        ("Conv2D_9", "Conv2D (6x6, 512 filters)"),
        ("ReLU_9", "ReLU"),

        ("Dropout", "Dropout (50%)"),
        ("FC_1", "Fully Connected (512 units)"),
        ("ReLU_10", "ReLU"),
        ("FC_2", "Fully Connected (8 units)"),
        ("Softmax", "Softmax"),
        ("Output", "Classification")
    ]

    # Add the nodes and edges
    for i, (layer, desc) in enumerate(layers):
        G.add_node(layer, desc=desc)
        if i > 0:
            G.add_edge(layers[i - 1][0], layer)

    # Set positions for the nodes
    pos = nx.spring_layout(G, seed=42)

    # Draw the nodes
    nx.draw(G, pos, with_labels=False, node_size=3000, node_color='skyblue', node_shape='s', alpha=0.9,
            edge_color='gray')

    # Draw the labels
    for layer, (x, y) in pos.items():
        plt.text(x, y, s=G.nodes[layer]['desc'], bbox=dict(facecolor='red', alpha=0.5), horizontalalignment='center',
                 fontsize=8)

    # Show the plot
    plt.show()


create_network_diagram()
