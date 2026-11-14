# visualize_network_graph.py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def create_network_graph():
    """Create a network graph visualization of the U-Net"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Define nodes with their properties
    nodes = [
        ('Input', {'layer': 'input', 'channels': 4, 'color': '#3498db', 'size': 3000}),
        ('E1', {'layer': 'encoder', 'channels': 16, 'color': '#2ecc71', 'size': 2500}),
        ('E2', {'layer': 'encoder', 'channels': 32, 'color': '#27ae60', 'size': 2000}),
        ('E3', {'layer': 'encoder', 'channels': 64, 'color': '#229954', 'size': 1500}),
        ('E4', {'layer': 'encoder', 'channels': 128, 'color': '#1e8449', 'size': 1000}),
        ('Bottleneck', {'layer': 'bottleneck', 'channels': 256, 'color': '#e74c3c', 'size': 800}),
        ('D4', {'layer': 'decoder', 'channels': 128, 'color': '#f39c12', 'size': 1000}),
        ('D3', {'layer': 'decoder', 'channels': 64, 'color': '#e67e22', 'size': 1500}),
        ('D2', {'layer': 'decoder', 'channels': 32, 'color': '#d35400', 'size': 2000}),
        ('D1', {'layer': 'decoder', 'channels': 16, 'color': '#ba4a00', 'size': 2500}),
        ('Output', {'layer': 'output', 'channels': 3, 'color': '#9b59b6', 'size': 3000})
    ]
    
    # Add nodes to graph
    for node, attrs in nodes:
        G.add_node(node, **attrs)
    
    # Add edges (forward path)
    forward_edges = [
        ('Input', 'E1'),
        ('E1', 'E2'),
        ('E2', 'E3'),
        ('E3', 'E4'),
        ('E4', 'Bottleneck'),
        ('Bottleneck', 'D4'),
        ('D4', 'D3'),
        ('D3', 'D2'),
        ('D2', 'D1'),
        ('D1', 'Output')
    ]
    
    # Add skip connections
    skip_edges = [
        ('E1', 'D1'),
        ('E2', 'D2'),
        ('E3', 'D3'),
        ('E4', 'D4')
    ]
    
    # Add all edges
    for edge in forward_edges:
        G.add_edge(*edge, color='black', style='solid', weight=2)
    
    for edge in skip_edges:
        G.add_edge(*edge, color='#34495e', style='dashed', weight=1)
    
    # Position nodes in U-shape
    pos = {
        'Input': (0, 5),
        'E1': (1, 4),
        'E2': (2, 3),
        'E3': (3, 2),
        'E4': (4, 1),
        'Bottleneck': (5, 0),
        'D4': (6, 1),
        'D3': (7, 2),
        'D2': (8, 3),
        'D1': (9, 4),
        'Output': (10, 5)
    }
    
    # Draw graph
    node_colors = [G.nodes[n]['color'] for n in G.nodes()]
    node_sizes = [G.nodes[n]['size'] for n in G.nodes()]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=node_sizes, alpha=0.8, ax=ax1)
    
    # Draw edges with different styles
    solid_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['style'] == 'solid']
    dashed_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['style'] == 'dashed']
    
    nx.draw_networkx_edges(G, pos, edgelist=solid_edges, 
                          edge_color='black', width=2, ax=ax1)
    nx.draw_networkx_edges(G, pos, edgelist=dashed_edges,
                          edge_color='#34495e', width=1, 
                          style='dashed', alpha=0.6, ax=ax1)
    
    # Add labels
    labels = {}
    for node in G.nodes():
        channels = G.nodes[node]['channels']
        if node == 'Input':
            labels[node] = f'Input\n{channels} channels'
        elif node == 'Output':
            labels[node] = f'Output\n{channels} tumor maps'
        elif node == 'Bottleneck':
            labels[node] = f'Bottleneck\n{channels} channels'
        else:
            labels[node] = f'{node}\n{channels} channels'
    
    nx.draw_networkx_labels(G, pos, labels, font_size=9, ax=ax1)
    
    ax1.set_title('U-Net Architecture Graph', fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # Create feature map visualization on second axis
    create_feature_map_visualization(ax2)
    
    plt.tight_layout()
    plt.show()

def create_feature_map_visualization(ax):
    """Create feature map size visualization"""
    
    # Feature map sizes through the network
    stages = ['Input', 'Enc1', 'Enc2', 'Enc3', 'Enc4', 'Bottleneck', 'Dec4', 'Dec3', 'Dec2', 'Dec1', 'Output']
    channels = [4, 16, 32, 64, 128, 256, 128, 64, 32, 16, 3]
    sizes = [128, 64, 32, 16, 8, 8, 16, 32, 64, 128, 128]  # Spatial size
    
    x = np.arange(len(stages))
    width = 0.35
    
    # Plot channel count
    ax.bar(x - width/2, channels, width, label='Channels', color='#3498db', alpha=0.7)
    
    # Plot spatial size
    ax2 = ax.twinx()
    ax2.bar(x + width/2, sizes, width, label='Spatial Size', color='#e74c3c', alpha=0.7)
    
    ax.set_xlabel('Network Stage', fontsize=12)
    ax.set_ylabel('Number of Channels', fontsize=12, color='#3498db')
    ax2.set_ylabel('Spatial Dimension', fontsize=12, color='#e74c3c')
    
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=45, ha='right')
    
    ax.set_title('Feature Map Dimensions Through Network', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

if __name__ == "__main__":
    create_network_graph()