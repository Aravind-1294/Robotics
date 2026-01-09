import numpy as np
import heapq

def get_global_path(grid, start, goal, resolution=1.0):
    w, h = grid.shape
    start_node = (int(start[0]/resolution), int(start[1]/resolution))
    goal_node = (int(goal[0]/resolution), int(goal[1]/resolution))
    
    start_node = (np.clip(start_node[0], 0, w-1), np.clip(start_node[1], 0, h-1))
    goal_node = (np.clip(goal_node[0], 0, w-1), np.clip(goal_node[1], 0, h-1))

    open_set = []
    heapq.heappush(open_set, (0, start_node))
    came_from = {}
    g_score = {start_node: 0}
    f_score = {start_node: np.linalg.norm(np.array(start_node) - np.array(goal_node))}
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if current == goal_node:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_node)
            path.reverse()
            world_path = [(x * resolution + resolution/2, y * resolution + resolution/2) for x, y in path]
            return world_path
            
        x, y = current
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1), 
                     (x+1, y+1), (x-1, y-1), (x+1, y-1), (x-1, y+1)]
        
        for nx, ny in neighbors:
            if 0 <= nx < w and 0 <= ny < h and grid[nx, ny] == 0:
                tentative_g = g_score[current] + np.linalg.norm(np.array((nx, ny)) - np.array((x, y)))
                neighbor = (nx, ny)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + np.linalg.norm(np.array(neighbor) - np.array(goal_node))
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    
    return []

class APFPlanner:
    def __init__(self, katt=5.0, krep=200.0, d0=4.0, robot_radius=2.2):
        self.katt = katt
        self.krep = krep
        self.d0 = d0
        self.robot_radius = robot_radius
        
        self.use_hybrid = False
        self.path = []
        self.waypoint_idx = 0
        self.waypoint_dist_threshold = 2.5
        self.current_target = None

    def set_hybrid_path(self, path):
        self.use_hybrid = True
        self.path = path
        self.waypoint_idx = 0

    def attraction_force(self, robot_pos, goal_pos):
        target = goal_pos
        
        if self.use_hybrid and self.path:
            if self.waypoint_idx < len(self.path):
                current_wp = np.array(self.path[self.waypoint_idx])
                dist = np.linalg.norm(robot_pos - current_wp)
                
                if dist < self.waypoint_dist_threshold:
                    self.waypoint_idx += 1
                    if self.waypoint_idx < len(self.path):
                        target = np.array(self.path[self.waypoint_idx])
                    else:
                        target = goal_pos
                else:
                    target = current_wp
            else:
                target = goal_pos
                
        self.current_target = target
        return -self.katt * (robot_pos - target)

    def repulsion_force(self, robot_pos, obstacles):
        """
        Calculates the total repulsive force by SUMMING forces from all obstacles.
        This creates smooth, curved trajectories.
        """
        f_rep = np.zeros(2)
        
        for obs in obstacles:
            ox, oy, w, h = obs
            
            closest_x = np.clip(robot_pos[0], ox, ox + w)
            closest_y = np.clip(robot_pos[1], oy, oy + h)
            closest_pt = np.array([closest_x, closest_y])
            
            dist_vec = robot_pos - closest_pt
            dist = np.linalg.norm(dist_vec)
            
            dist_surface = dist - self.robot_radius
            
            if dist_surface < self.d0:
                if dist_surface <= 0.001:
                    dist_surface = 0.001
                    
                f_mag = self.krep * (1.0 / (dist_surface**2))
                
                if dist > 0:
                    f_dir = dist_vec / dist
                else:
                    f_dir = np.array([1.0, 0.0])
                    
                f_rep += f_mag * f_dir

        mag = np.linalg.norm(f_rep)
        if mag > 500.0:
            f_rep = (f_rep / mag) * 500.0
            
        return f_rep

    def get_potential(self, robot_pos, goal_pos, obstacles):
        dist_goal = np.linalg.norm(robot_pos - goal_pos)
        u_att = 0.5 * self.katt * (dist_goal**2)
        
        u_rep = 0.0
        for obs in obstacles:
            ox, oy, w, h = obs
            closest_x = np.clip(robot_pos[0], ox, ox + w)
            closest_y = np.clip(robot_pos[1], oy, oy + h)
            dist = np.linalg.norm(robot_pos - np.array([closest_x, closest_y]))
            
            if dist < self.d0:
                if dist <= 0.001: dist = 0.001
                u_rep += self.krep * (1.0 / dist - 1.0 / self.d0)
                
        return u_att + u_rep

    def get_total_force(self, robot_pos, goal_pos, obstacles):
        f_att = self.attraction_force(robot_pos, goal_pos)
        f_rep = self.repulsion_force(robot_pos, obstacles)
        return f_att + f_rep
