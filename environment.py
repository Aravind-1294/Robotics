import numpy as np

class GridEnvironment:
    def __init__(self, width=30, height=30):
        self.width = width
        self.height = height
        self.obstacles = []

class RectangularRobot:
    def __init__(self, position, velocity, size):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.size = size

def load_scenario(name):
    obstacles = []
    start = np.array([5.0, 5.0])
    goal = np.array([20.0, 20.0])
    
    obstacles.append((0, 0, 0.5, 30))
    obstacles.append((29.5, 0, 0.5, 30))
    obstacles.append((0, 0, 30, 0.5))
    obstacles.append((0, 29.5, 30, 0.5))

    if name == 'Empty':
        pass

    elif name == 'Single':
        obstacles.append((9, 15, 2, 2))
        obstacles.append((16, 8, 2, 2))

    elif name == 'Complex':
        obstacles.append((10, 12, 3, 3))
        obstacles.append((16, 5, 2, 2))
        obstacles.append((13, 16, 2, 2))
        obstacles.append((22, 12, 2, 2))

    elif name == 'Corridor':
        obstacles.append((0, 10, 20, 2))
        obstacles.append((10, 18, 20, 2))

    elif name == 'U-Trap':
        start = np.array([4.0, 4.0])
        goal = np.array([26.0, 26.0])
        obstacles.append((18, 2, 2, 20))
        obstacles.append((10, 20, 8, 2))
        obstacles.append((10, 2, 8, 2))

    else:
        raise ValueError(f"Unknown scenario: {name}")

    return obstacles, start, goal

def create_occupancy_grid(width, height, obstacles, resolution=1.0):
    grid_w = int(width / resolution)
    grid_h = int(height / resolution)
    grid = np.zeros((grid_w, grid_h), dtype=int)
    
    margin = 2.5
    
    for obs in obstacles:
        ox, oy, w, h = obs
        gx_min = int((ox - margin) / resolution)
        gx_max = int((ox + w + margin) / resolution)
        gy_min = int((oy - margin) / resolution)
        gy_max = int((oy + h + margin) / resolution)
        
        gx_min = max(0, gx_min)
        gx_max = min(grid_w, gx_max)
        gy_min = max(0, gy_min)
        gy_max = min(grid_h, gy_max)
        
        grid[gx_min:gx_max, gy_min:gy_max] = 1
        
    return grid

def check_collision(robot_pos, robot_size, obstacles):
    rx, ry = robot_pos
    rw, rh = robot_size
    
    for obs in obstacles:
        ox, oy, ow, oh = obs
        
        if (rx < ox + ow and
            rx + rw > ox and
            ry < oy + oh and
            ry + rh > oy):
            return True
            
    return False
