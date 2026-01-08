import numpy as np
import time
from environment import RectangularRobot, load_scenario, create_occupancy_grid
from planner import APFPlanner, get_global_path

def run_gif_sim(scenario_name, use_hybrid=False, gif_name="sim.gif"):
    obstacles, start, goal = load_scenario(scenario_name)
    
    robot = RectangularRobot(position=start, velocity=[0,0], size=(3.0, 2.0)) 
    
    local_d0 = 2.0 if use_hybrid else 4.0
    planner = APFPlanner(katt=0.5, krep=200.0, d0=local_d0) 
    planner.waypoint_dist_threshold = 2.5 
    
    if use_hybrid:
        grid_map = create_occupancy_grid(30, 30, obstacles, resolution=1.0)
        path = get_global_path(grid_map, start, goal, resolution=1.0)
        if path:
            planner.set_hybrid_path(path)
            
    max_frames = 600
    steps_per_frame = 20
    sim_dt = 0.005
    dt_frame = 0.1
    
    success = False
    frames_taken = 0
    
    for frame in range(max_frames):
        for _ in range(steps_per_frame):
            total_force = planner.get_total_force(robot.position, goal, obstacles)
            robot.position = robot.position + total_force * sim_dt
            
            if np.linalg.norm(robot.position - goal) < 2.0:
                success = True
                break
        
        frames_taken += 1
        if success:
            break
            
    time_taken = frames_taken * dt_frame
    total_physics_steps = frames_taken * steps_per_frame
    
    if not success:
        dist = np.linalg.norm(robot.position - goal)
        print(f"DEBUG: {scenario_name} {use_hybrid} Failed. Dist: {dist:.4f}")

    return {
        "GIF": gif_name,
        "Scenario": scenario_name,
        "Mode": "Hybrid" if use_hybrid else "APF",
        "Success": success,
        "Time (s)": f"{time_taken:.1f}",
        "Steps": total_physics_steps
    }

def main():
    configs = [
        ("U-Trap", False, "apf_failure.gif"),
        ("U-Trap", True,  "hybrid_success.gif"),
        ("Empty",  False, "apf_success_empty.gif"),
        ("Single", False, "apf_success_obstacles.gif"),
        ("Complex", False, "apf_success_complex.gif")
    ]
    
    print(f"{'GIF Name':<25} | {'Scenario':<10} | {'Mode':<8} | {'Success':<8} | {'Time(s)':<8} | {'Steps':<8}")
    print("-" * 85)
    
    for scen, hybrid, gif in configs:
        res = run_gif_sim(scen, hybrid, gif)
        print(f"{res['GIF']:<25} | {res['Scenario']:<10} | {res['Mode']:<8} | {str(res['Success']):<8} | {res['Time (s)']:<8} | {res['Steps']:<8}")

if __name__ == "__main__":
    main()
