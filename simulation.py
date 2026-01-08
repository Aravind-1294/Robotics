import numpy as np
from environment import RectangularRobot, load_scenario, create_occupancy_grid, check_collision
from planner import APFPlanner, get_global_path

def run_headless_simulation(scenario_name, max_steps=1000, use_hybrid=False):
    obstacles, start, goal = load_scenario(scenario_name)
    
    robot = RectangularRobot(position=start, velocity=[0,0], size=(3.0, 2.0))
    planner = APFPlanner(katt=0.5, krep=200.0, d0=4.0) 
    planner.waypoint_dist_threshold = 2.5
    
    if use_hybrid:
        grid_map = create_occupancy_grid(30, 30, obstacles, resolution=1.0)
        path = get_global_path(grid_map, start, goal, resolution=1.0)
        if path:
            planner.set_hybrid_path(path)
        else:
            pass

    dt = 0.1
    sim_dt = 0.005
    steps_per_frame = 20
    
    path_length = 0.0
    steps_taken = 0
    success = False
    min_clearance = float('inf')
    
    for step in range(max_steps):
        for _ in range(steps_per_frame):
            prev_pos = robot.position.copy()
            
            total_force = planner.get_total_force(robot.position, goal, obstacles)
            
            robot.position = robot.position + total_force * sim_dt
            
            dist_step = np.linalg.norm(robot.position - prev_pos)
            path_length += dist_step
            
            for obs in obstacles:
                ox, oy, w, h = obs
                closest_x = np.clip(robot.position[0], ox, ox + w)
                closest_y = np.clip(robot.position[1], oy, oy + h)
                dist = np.linalg.norm(robot.position - np.array([closest_x, closest_y]))
                if dist < min_clearance:
                    min_clearance = dist
            
            if check_collision(robot.position, robot.size, obstacles):
                break
                
            dist_to_goal = np.linalg.norm(robot.position - goal)
            if dist_to_goal < 1.0:
                success = True
                break
        
        steps_taken += 1
        if check_collision(robot.position, robot.size, obstacles):
            break
        if success:
            break
            
    return {
        'scenario': scenario_name,
        'mode': 'Hybrid' if use_hybrid else 'APF',
        'success': success,
        'steps': steps_taken,
        'length': path_length,
        'min_clearance': min_clearance
    }

if __name__ == "__main__":
    run_headless_simulation('U-Trap', max_steps=1000, use_hybrid=True)
