import numpy as np
import matplotlib.pyplot as plt
import os

# Configure English fonts
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def create_demo():
    # Create output directory
    output_dir = './demo_outputs/'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Qianfan Satellite Demo - English Version")
    print("="*40)
    
    # Generate sample data
    t = np.linspace(0, 3600, 1000)
    distance = 100 * np.exp(-t/1800)
    x = distance * np.cos(0.1 * np.sin(t/600))
    y = distance * 0.3 * np.sin(t/800)
    z = distance * 0.2 * np.cos(t/1000)
    
    # Plot 3D trajectory
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(x, y, z, c=t/60, cmap='viridis', s=10, alpha=0.7)
    ax.scatter([x[0]], [y[0]], [z[0]], color='red', s=100, marker='o', label='Start')
    ax.scatter([0], [0], [0], color='gold', s=200, marker='*', label='Target')
    
    ax.set_xlabel('Radial Distance (m)')
    ax.set_ylabel('Along-track Distance (m)')
    ax.set_zlabel('Cross-track Distance (m)')
    ax.set_title('Qianfan Satellite 3D Trajectory', fontweight='bold')
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
    cbar.set_label('Time (minutes)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, '3d_trajectory.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"3D trajectory saved: {save_path}")
    plt.close()
    
    # Plot performance
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    t_min = t / 60
    total_distance = np.sqrt(x**2 + y**2 + z**2)
    vx, vy, vz = np.gradient(x, t), np.gradient(y, t), np.gradient(z, t)
    velocity = np.sqrt(vx**2 + vy**2 + vz**2)
    
    # Distance plot
    ax1.plot(t_min, total_distance, 'b-', linewidth=2, label='Distance to Target')
    ax1.axhline(y=10.0, color='orange', linestyle='--', label='Safety Zone')
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Distance (m)')
    ax1.set_title('Distance Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Velocity plot
    ax2.plot(t_min, velocity * 1000, 'g-', linewidth=2, label='Velocity')
    ax2.axhline(y=100, color='red', linestyle='--', label='Speed Limit')
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Velocity (mm/s)')
    ax2.set_title('Velocity Control')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Thrust plot
    thrust_x = -0.01 * x - 0.1 * vx
    thrust_y = -0.01 * y - 0.1 * vy
    thrust_z = -0.01 * z - 0.1 * vz
    
    ax3.plot(t_min, thrust_x * 1000, 'r-', label='Radial Thrust')
    ax3.plot(t_min, thrust_y * 1000, 'g-', label='Along-track Thrust')
    ax3.plot(t_min, thrust_z * 1000, 'b-', label='Cross-track Thrust')
    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylabel('Thrust (mN)')
    ax3.set_title('Thrust Control')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Phase plot
    ax4.scatter(x, vx * 1000, c=t_min, cmap='plasma', s=15, alpha=0.7)
    ax4.set_xlabel('Position (m)')
    ax4.set_ylabel('Velocity (mm/s)')
    ax4.set_title('Phase Space')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Qianfan Satellite Control Analysis', fontweight='bold')
    plt.tight_layout()
    
    save_path2 = os.path.join(output_dir, 'control_performance.png')
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"Control performance saved: {save_path2}")
    plt.close()
    
    print("Demo completed successfully!")
    return [save_path, save_path2]

if __name__ == "__main__":
    create_demo()
