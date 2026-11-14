# visualize_3d_architecture.py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def create_3d_unet_diagram():
    """Create a 3D visualization of the U-Net architecture"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Create 3D axis
    ax = fig.add_subplot(111, projection='3d')
    
    # Colors for different parts
    colors = {
        'input': '#3498db',      # Blue
        'encoder': '#2ecc71',    # Green
        'bottleneck': '#e74c3c', # Red
        'decoder': '#f39c12',    # Orange
        'output': '#9b59b6',     # Purple
        'skip': '#34495e'        # Dark gray
    }
    
    # U-Net dimensions (depth decreases then increases)
    levels = 5
    max_channels = 256
    max_size = 10
    
    # Create encoder (left side - decreasing size)
    encoder_boxes = []
    for i in range(levels):
        size = max_size * (0.8 ** i)
        channels = 16 * (2 ** i)
        
        # Create cube
        x = -i * 3  # Move left
        y = 0
        z = 0
        
        # Add cube
        cube = create_cube(x, y, z, size)
        encoder_boxes.append({
            'cube': cube,
            'position': (x, y, z),
            'size': size,
            'channels': channels,
            'label': f'Encoder {i+1}\n{channels} channels'
        })
    
    # Create bottleneck (bottom)
    bottleneck = {
        'cube': create_cube(-(levels-1)*3, 0, -4, max_size*0.4),
        'position': (-(levels-1)*3, 0, -4),
        'size': max_size*0.4,
        'channels': 256,
        'label': 'Bottleneck\n256 channels'
    }
    
    # Create decoder (right side - increasing size)
    decoder_boxes = []
    for i in range(levels):
        j = levels - i - 1  # Reverse order
        size = max_size * (0.8 ** j)
        channels = 16 * (2 ** j)
        
        # Create cube
        x = -j * 3  # Symmetric with encoder
        y = 0
        z = 0
        
        # Add cube
        cube = create_cube(x, y, z, size)
        decoder_boxes.append({
            'cube': cube,
            'position': (x, y, z),
            'size': size,
            'channels': channels,
            'label': f'Decoder {i+1}\n{channels} channels'
        })
    
    # Plot encoder boxes (green)
    for box in encoder_boxes:
        poly = Poly3DCollection(box['cube'], alpha=0.7, 
                               facecolors=colors['encoder'], 
                               edgecolors='black', linewidths=1)
        ax.add_collection3d(poly)
        
        # Add label
        ax.text(box['position'][0], box['position'][1], 
               box['position'][2] + box['size']/2, 
               box['label'], ha='center', fontsize=8)
    
    # Plot bottleneck (red)
    poly = Poly3DCollection(bottleneck['cube'], alpha=0.9,
                           facecolors=colors['bottleneck'],
                           edgecolors='black', linewidths=2)
    ax.add_collection3d(poly)
    ax.text(bottleneck['position'][0], bottleneck['position'][1],
           bottleneck['position'][2] + bottleneck['size']/2,
           bottleneck['label'], ha='center', fontsize=9, fontweight='bold')
    
    # Plot decoder boxes (orange)
    for box in decoder_boxes:
        poly = Poly3DCollection(box['cube'], alpha=0.7,
                               facecolors=colors['decoder'],
                               edgecolors='black', linewidths=1)
        ax.add_collection3d(poly)
        
        # Add label
        ax.text(box['position'][0], box['position'][1],
               box['position'][2] + box['size']/2,
               box['label'], ha='center', fontsize=8)
    
    # Add skip connections (gray lines)
    for i in range(levels-1):
        # From encoder to decoder
        start = encoder_boxes[i]['position']
        end = decoder_boxes[levels-i-2]['position']
        
        ax.plot([start[0], end[0]], [start[1], end[1]], 
                [start[2] + encoder_boxes[i]['size']/2, 
                 end[2] + decoder_boxes[levels-i-2]['size']/2],
                color=colors['skip'], linewidth=2, linestyle='--', alpha=0.6)
    
    # Add input and output
    input_box = create_cube(3, 0, 0, max_size)
    output_box = create_cube(-(levels-1)*3 - 3, 0, 0, max_size*0.8)
    
    # Plot input
    poly = Poly3DCollection(input_box, alpha=0.8,
                           facecolors=colors['input'],
                           edgecolors='black', linewidths=2)
    ax.add_collection3d(poly)
    ax.text(3, 0, max_size/2, 'Input\n4 MRI channels', 
           ha='center', fontsize=9, fontweight='bold')
    
    # Plot output
    poly = Poly3DCollection(output_box, alpha=0.8,
                           facecolors=colors['output'],
                           edgecolors='black', linewidths=2)
    ax.add_collection3d(poly)
    ax.text(-(levels-1)*3 - 3, 0, max_size*0.4,
           'Output\n3 Tumor maps', ha='center', fontsize=9, fontweight='bold')
    
    # Set plot limits and labels
    ax.set_xlim(-15, 10)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 10)
    
    ax.set_xlabel('Network Depth')
    ax.set_ylabel('Width')
    ax.set_zlabel('Height')
    ax.set_title('3D U-Net Architecture Visualization', fontsize=16, fontweight='bold')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color=colors['input'], lw=4, label='Input Layer'),
        plt.Line2D([0], [0], color=colors['encoder'], lw=4, label='Encoder'),
        plt.Line2D([0], [0], color=colors['bottleneck'], lw=4, label='Bottleneck'),
        plt.Line2D([0], [0], color=colors['decoder'], lw=4, label='Decoder'),
        plt.Line2D([0], [0], color=colors['output'], lw=4, label='Output Layer'),
        plt.Line2D([0], [0], color=colors['skip'], lw=2, linestyle='--', label='Skip Connections')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()

def create_cube(x, y, z, size):
    """Create vertices for a cube"""
    vertices = np.array([
        [x-size/2, y-size/2, z-size/2],
        [x+size/2, y-size/2, z-size/2],
        [x+size/2, y+size/2, z-size/2],
        [x-size/2, y+size/2, z-size/2],
        [x-size/2, y-size/2, z+size/2],
        [x+size/2, y-size/2, z+size/2],
        [x+size/2, y+size/2, z+size/2],
        [x-size/2, y+size/2, z+size/2]
    ])
    
    # Define the 6 faces of the cube
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
        [vertices[1], vertices[2], vertices[6], vertices[5]]   # right
    ]
    
    return faces

if __name__ == "__main__":
    create_3d_unet_diagram()