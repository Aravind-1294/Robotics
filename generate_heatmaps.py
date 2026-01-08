import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
from environment import load_scenario
from planner import APFPlanner

def generate_heatmap(scenario_name, filename):
    print(f"Generating Force Magnitude Heatmap for {scenario_name}...")
    
    obstacles, start, goal = load_scenario(scenario_name)
    planner = APFPlanner(katt=0.5, krep=200.0, d0=4.0)
    
    resolution = 0.2
    width, height = 30, 30
    x_vals = np.arange(0, width, resolution)
    y_vals = np.arange(0, height, resolution)
    X, Y = np.meshgrid(x_vals, y_vals)
    force_map = np.zeros_like(X)
    
    rows, cols = X.shape
    for i in range(rows):
        for j in range(cols):
            pos = np.array([X[i, j], Y[i, j]])
            f_vec = planner.get_total_force(pos, goal, obstacles)
            force_map[i, j] = np.linalg.norm(f_vec)

    fig, ax = plt.subplots(figsize=(10, 8))
    
    print(f"  Range: Min={np.min(force_map):.2f}, Max={np.max(force_map):.2f}")

    f_goal = np.linalg.norm(planner.get_total_force(goal, goal, obstacles))
    f_trap = np.linalg.norm(planner.get_total_force(np.array([12.0, 12.0]), goal, obstacles))
    print(f"  Probe: Goal(20,20)={f_goal:.2f}, TrapApprox(12,12)={f_trap:.2f}")

    display_map = np.clip(force_map, 0, 50.0)
    
    mesh = ax.imshow(display_map, origin='lower', extent=[0, width, 0, height],
                     cmap='jet', vmin=0.0, vmax=50.0, alpha=0.9)
    
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label('Force Magnitude (Linear Scale 0-50)')
    
    for obs in obstacles:
        ox, oy, w, h = obs
        rect = patches.Rectangle((ox, oy), w, h, linewidth=1, edgecolor='black', facecolor='gray')
        ax.add_patch(rect)
        
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start', markeredgecolor='black')
    ax.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal', markeredgecolor='black')
    
    ax.set_title(f'Force Magnitude Landscape: {scenario_name}')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='upper left')
    ax.grid(False)
    
    plt.tight_layout()
    output_path = os.path.join("results", filename)
    plt.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")
    plt.close(fig)

if __name__ == "__main__":
    scenarios = [
        ('U-Trap', 'heatmap_force_U-Trap.png')
    ]
    
    for name, fname in scenarios:
        generate_heatmap(name, fname)
