import numpy as np
from simulation import load_scenario, APFPlanner, create_occupancy_grid, check_collision, RectangularRobot

def diagnose():
    print("--- DIAGNOSTIC START ---")
    obstacles, start, goal = load_scenario('U-Trap')
    print(f"Start: {start}")
    print(f"Goal: {goal}")
    print(f"Obstacles ({len(obstacles)}):")
    for o in obstacles:
        print(f"  {o}")

    robot_size = (2,2)
    radius = 1.5
    if check_collision(start, robot_size, obstacles):
        print("CRITICAL: Robot spawns INSIDE an obstacle!")
    else:
        print("Spawn check: OK (No direct collision)")

    planner = APFPlanner(katt=2.0, krep=20.0, d0=3.0)
    

    total_rep = np.zeros(2)
    for i, obs in enumerate(obstacles):
        ox, oy, w, h = obs

        closest_x = np.clip(start[0], ox, ox + w)
        closest_y = np.clip(start[1], oy, oy + h)
        closest_pt = np.array([closest_x, closest_y])
        dist_vec = start - closest_pt
        dist = np.linalg.norm(dist_vec)
        dist_surf = dist - radius
        
        if dist_surf < 3.0:
            f_rep_mag = 0
            if dist_surf <= 0.001:
                print(f"  Obs #{i} {obs}: CONTACT/INSIDE! dist_surf={dist_surf}")
            else:
                 f_rep_mag = 20.0 * (1.0 / (dist_surf**2))
                 print(f"  Obs #{i} {obs}: dist={dist:.2f}, surf={dist_surf:.2f}, F_rep_mag={f_rep_mag:.2f}")

    f_att = planner.attraction_force(start, goal)
    f_rep = planner.repulsion_force(start, obstacles)
    f_tot = f_att + f_rep
    
    print(f"Force Att: {f_att}")
    print(f"Force Rep: {f_rep}")
    print(f"Force Tot: {f_tot}")
    print(f"Force Mag: {np.linalg.norm(f_tot)}")

    print("--- A* Grid Check ---")
    grid = create_occupancy_grid(30, 30, obstacles, resolution=1.0)
    sx, sy = int(start[0]), int(start[1])
    print(f"Grid Start Node ({sx}, {sy}): Value={grid[sx, sy]}")
    
    for dx in [-1,0,1]:
        for dy in [-1,0,1]:
            nx, ny = sx+dx, sy+dy
            if 0<=nx<30 and 0<=ny<30:
                print(f"  ({nx},{ny}): {grid[nx,ny]}")

if __name__ == "__main__":
    diagnose()
