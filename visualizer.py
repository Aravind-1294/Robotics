import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from environment import GridEnvironment, RectangularRobot, load_scenario, create_occupancy_grid, check_collision
from planner import APFPlanner, get_global_path

def generate_video(filename, use_hybrid=False, scenario_name='U-Trap'):
    print(f"--- Generating Video: {filename} (Hybrid={use_hybrid}) ---")
    
    obstacles, start, goal = load_scenario(scenario_name)
    robot = RectangularRobot(position=start, velocity=[0,0], size=(3.0, 2.0))
    planner = APFPlanner(katt=0.5, krep=200.0, d0=4.0) 
    planner.waypoint_dist_threshold = 2.5 
    
    path = None
    if use_hybrid:
        print("Planning Hybrid Path...")
        grid_map = create_occupancy_grid(30, 30, obstacles, resolution=1.0)
        path = get_global_path(grid_map, start, goal, resolution=1.0)
        if path:
            planner.set_hybrid_path(path)
            print(f"Path found: {len(path)} waypoints")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.set_aspect('equal')
    
    mode_str = "Hybrid Success" if use_hybrid else "Standard APF Failure"
    ax.set_title(f"{mode_str}: {scenario_name}")
    ax.grid(True, linestyle='--', alpha=0.5)

    for obs in obstacles:
        ox, oy, w, h = obs
        rect = patches.Rectangle((ox, oy), w, h, linewidth=1, edgecolor='black', facecolor='gray')
        ax.add_patch(rect)

    ax.plot(start[0], start[1], 'go', markersize=8, label='Start')
    ax.plot(goal[0], goal[1], 'rx', markersize=8, label='Goal')

    if path:
        px, py = zip(*path)
        ax.plot(px, py, 'g--', alpha=0.5, label='A* Path')

    robot_patch = patches.Rectangle((start[0], start[1]), robot.size[0], robot.size[1], 
                                    linewidth=1, edgecolor='blue', facecolor='cyan', alpha=0.8)
    ax.add_patch(robot_patch)
    
    trail_x, trail_y = [start[0]], [start[1]]
    trail_line, = ax.plot([], [], 'b-', linewidth=1, alpha=0.6)
    
    target_marker, = ax.plot([], [], 'D', color='orange', markersize=6, label='Current Target')
    
    metrics_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    dt = 0.1 
    sim_dt = 0.005 
    steps_per_frame = 20 
    max_frames = 600
    
    def update(frame):
        nonlocal trail_x, trail_y
        
        if np.linalg.norm(robot.position - goal) < 1.0:
             metrics_text.set_text("GOAL REACHED!")
             return robot_patch, trail_line, target_marker, metrics_text
        
        for _ in range(steps_per_frame):
            total_force = planner.get_total_force(robot.position, goal, obstacles)
            robot.position = robot.position + total_force * sim_dt
            
            if np.linalg.norm(robot.position - goal) < 1.0:
                metrics_text.set_text("GOAL REACHED!")
                return robot_patch, trail_line, target_marker, metrics_text
        
        robot_patch.set_xy((robot.position[0], robot.position[1]))
        
        trail_x.append(robot.position[0])
        trail_y.append(robot.position[1])
        trail_line.set_data(trail_x, trail_y)
        
        if planner.current_target is not None:
            target_marker.set_data([planner.current_target[0]], [planner.current_target[1]])
        else:
            target_marker.set_data([goal[0]], [goal[1]])
            
        vel_mag = np.linalg.norm(total_force)
        metrics_text.set_text(f"Force Mag: {vel_mag:.2f}")
        
        return robot_patch, trail_line, target_marker, metrics_text

    anim = FuncAnimation(fig, update, frames=max_frames, interval=100, blit=True)
    anim.save(filename, writer='pillow', fps=10)
    print(f"Video saved as {filename}")
    plt.close(fig)

if __name__ == "__main__":
    generate_video('apf_failure.gif', use_hybrid=False, scenario_name='U-Trap')
    generate_video('hybrid_success.gif', use_hybrid=True, scenario_name='U-Trap')
    generate_video('apf_success_empty.gif', use_hybrid=False, scenario_name='Empty')
    generate_video('apf_success_obstacles.gif', use_hybrid=False, scenario_name='Single')
    generate_video('apf_success_complex.gif', use_hybrid=False, scenario_name='Complex')
