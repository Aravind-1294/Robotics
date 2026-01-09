import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from environment import RectangularRobot, create_occupancy_grid
from planner import APFPlanner, get_global_path

WIDTH = 30
HEIGHT = 30
START = np.array([4.0, 4.0])
GOAL = np.array([26.0, 26.0])
ROBOT_SIZE = (3.0, 2.0)

class InteractiveSim:
    def __init__(self):
        self.obstacles = []
        self.robot = RectangularRobot(START, [0,0], ROBOT_SIZE)
        radius = np.linalg.norm(np.array(ROBOT_SIZE)) / 2.0
        
        # Default parameters
        self.katt = 0.5
        self.krep = 50.0
        self.d0 = 2.0
        
        self.planner = APFPlanner(katt=self.katt, krep=self.krep, d0=self.d0, robot_radius=radius)
        self.planner.waypoint_dist_threshold = 2.5
        self.running = False
        self.finished = False
        self.mode = "APF"
        self.path = None
        
        # Setup Figure with sliders
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111)
        
        self.setup_plot()
        self.setup_sliders()
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.anim = FuncAnimation(self.fig, self.update, interval=50, blit=False)
        plt.show()

    def setup_plot(self):
        self.ax.set_xlim(0, WIDTH)
        self.ax.set_ylim(0, HEIGHT)
        self.ax.set_aspect('equal')
        self.ax.set_xticks(np.arange(0, WIDTH+1, 5))
        self.ax.set_yticks(np.arange(0, HEIGHT+1, 5))
        self.ax.grid(True, linestyle='-', alpha=0.3)
        self.ax.set_title("Interactive Mode\nLeft Click: Add Obstacle | 'm': Toggle Mode | 'SPACE': Start | 'r': Reset")
        self.ax.plot(START[0], START[1], 'go', markersize=8)
        self.ax.plot(GOAL[0], GOAL[1], 'rx', markersize=8)
        self.robot_patch = patches.Rectangle((START[0], START[1]), ROBOT_SIZE[0], ROBOT_SIZE[1], color='cyan', alpha=0.8)
        self.ax.add_patch(self.robot_patch)
        self.obstacle_patches = []
        self.path_line, = self.ax.plot([], [], 'g--', alpha=0.5, linewidth=2)
        self.mode_text = self.ax.text(0.02, 0.95, f"Mode: {self.mode}", transform=self.ax.transAxes, fontweight='bold')
        self.status_text = self.ax.text(0.02, 0.90, "Status: PAUSED", transform=self.ax.transAxes, color='red')

    def setup_sliders(self):
        # Adjust main plot to make room for 3 sliders
        self.ax.set_position([0.1, 0.35, 0.8, 0.6])
        
        # Create slider axes
        ax_katt = plt.axes([0.15, 0.25, 0.7, 0.03])
        ax_krep = plt.axes([0.15, 0.18, 0.7, 0.03])
        ax_d0 = plt.axes([0.15, 0.11, 0.7, 0.03])
        
        # Create sliders
        self.slider_katt = Slider(ax_katt, 'katt (Attraction)', 0.1, 3.0, valinit=self.katt, valstep=0.1)
        self.slider_krep = Slider(ax_krep, 'krep (Repulsion)', 10.0, 200.0, valinit=self.krep, valstep=10.0)
        self.slider_d0 = Slider(ax_d0, 'd0 (Range)', 0.5, 5.0, valinit=self.d0, valstep=0.5)
        
        # Connect sliders to update function
        self.slider_katt.on_changed(self.update_params)
        self.slider_krep.on_changed(self.update_params)
        self.slider_d0.on_changed(self.update_params)

    def update_params(self, val):
        if not self.running:
            self.katt = self.slider_katt.val
            self.krep = self.slider_krep.val
            self.d0 = self.slider_d0.val
            self.planner.katt = self.katt
            self.planner.krep = self.krep
            self.planner.d0 = self.d0
            print(f"Updated: katt={self.katt:.1f}, krep={self.krep:.1f}, d0={self.d0:.1f}")

    def on_click(self, event):
        if event.inaxes != self.ax: return
        if self.running: return
        w, h = 2.5, 2.5
        ox = int(event.xdata / 2.5) * 2.5
        oy = int(event.ydata / 2.5) * 2.5
        self.obstacles.append((ox, oy, w, h))
        rect = patches.Rectangle((ox, oy), w, h, linewidth=1, edgecolor='black', facecolor='gray')
        self.ax.add_patch(rect)
        self.obstacle_patches.append(rect)
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'm':
            if not self.running:
                if self.mode == "APF":
                    self.mode = "Hybrid"
                elif self.mode == "Hybrid":
                    self.mode = "A* Only"
                else:
                    self.mode = "APF"
                self.mode_text.set_text(f"Mode: {self.mode}")
                self.fig.canvas.draw()
        elif event.key == ' ':
            if not self.running and not self.finished:
                self.start_simulation()
        elif event.key == 'r':
            self.reset()

    def start_simulation(self):
        print("Starting Simulation...")
        self.running = True
        self.status_text.set_text("Status: RUNNING")
        self.status_text.set_color('green')
        if self.mode == "Hybrid" or self.mode == "A* Only":
            print("Calculating A* Path...")
            grid = create_occupancy_grid(WIDTH, HEIGHT, self.obstacles, resolution=0.5)
            self.path = get_global_path(grid, START, GOAL, resolution=0.5)
            if self.path:
                print(f"Path Found: {len(self.path)} steps")
                if self.mode == "Hybrid":
                    self.planner.set_hybrid_path(self.path)
                    self.planner.krep = 20.0
                px, py = zip(*self.path)
                self.path_line.set_data(px, py)
            else:
                print("No A* Path Found!")
                self.status_text.set_text("Status: NO PATH!")
        else:
            self.planner.use_hybrid = False
            self.planner.krep = self.krep
            self.path_line.set_data([], [])

    def reset(self):
        self.running = False
        self.finished = False
        self.robot.position = START.copy()
        self.robot_patch.set_xy(START)
        for p in self.obstacle_patches: p.remove()
        self.obstacle_patches = []
        self.obstacles = []
        self.path_line.set_data([], [])
        self.status_text.set_text("Status: RESET")
        self.fig.canvas.draw()

    def update(self, frame):
        if not self.running or self.finished: return
        center_pos = self.robot.position + np.array(ROBOT_SIZE) / 2.0
        if np.linalg.norm(center_pos - GOAL) < 1.5:
            self.finished = True
            self.status_text.set_text("Status: SUCCESS!")
            self.status_text.set_color('blue')
            return
        for _ in range(20):
            if self.mode == "A* Only" and self.path:
                if len(self.path) > 0:
                    target = np.array(self.path[0])
                    direction = target - self.robot.position
                    dist = np.linalg.norm(direction)
                    if dist < 0.5:
                        self.path.pop(0)
                        if len(self.path) == 0:
                            break
                    else:
                        self.robot.position += (direction / dist) * 0.1
            else:
                center_pos = self.robot.position + np.array(ROBOT_SIZE) / 2.0
                force = self.planner.get_total_force(center_pos, GOAL, self.obstacles)
                new_center = center_pos + force * 0.005
                self.robot.position = new_center - np.array(ROBOT_SIZE) / 2.0
            center_pos = self.robot.position + np.array(ROBOT_SIZE) / 2.0
            if np.linalg.norm(center_pos - GOAL) < 1.5:
                self.finished = True
                self.status_text.set_text("Status: SUCCESS!")
                self.status_text.set_color('blue')
                break
        self.robot_patch.set_xy(self.robot.position)
        return self.robot_patch,

if __name__ == "__main__":
    sim = InteractiveSim()
