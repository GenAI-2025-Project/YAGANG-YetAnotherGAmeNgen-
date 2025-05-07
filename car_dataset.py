import os, random, torch, cv2, pygame, numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
from gym import spaces
from IPython.display import clear_output

# Output Schema Globals
GAME_NAME = "<car>"
FRAME_SIZE = (512, 512, 3)
TORCH_DATA_TYPE = torch.float32
STANDARD_GREY_IMAGE_TENSOR = torch.tensor(np.full(FRAME_SIZE, 128, dtype=np.uint8), dtype=torch.float32) / 127.5 - 1.0

def normalize_frame(frame):
    frame = cv2.resize(frame, (FRAME_SIZE[1], FRAME_SIZE[0]))
    return torch.tensor((frame.astype(np.float32) / 127.5 - 1.0), dtype=TORCH_DATA_TYPE)

def denormalize_frame(tensor):
    return ((tensor.numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)

# --- [2] CarGame Env ---
class CarGame:
    def __init__(self, width=600, height=600, lane_width=200, speed=10):
        self.width, self.height = width, height
        self.lane_width = lane_width
        self.speed = speed
        self.car_size, self.obstacle_size = 60, 60
        self.num_lanes, self.movement_speed = 3, 40
        self.obstacle_pattern = [1, 2, 3, 2]
        self.obstacle_gap = 200
        pygame.init()
        self.surface = pygame.Surface((width, height))
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.car_lane, self.car_y = 1, self.height - 100
        self.car_x = self.lane_width // 2 + self.car_lane * self.lane_width
        self.road_pos = 0
        self.pattern_index = random.randint(0, len(self.obstacle_pattern) - 1)
        self.obstacles, self.passed_obstacles = [], set()
        self._generate_initial_obstacles()
        self.score, self.steps, self.game_over = 0, 0, False
        return self._get_state()

    def _generate_initial_obstacles(self):
        for i in range(8):
            lane = (self.obstacle_pattern[self.pattern_index] - 1) % self.num_lanes
            self.obstacles.append({'lane': lane, 'y': -i * self.obstacle_gap, 'size': self.obstacle_size, 'id': i})
            self.pattern_index = (self.pattern_index + 1) % len(self.obstacle_pattern)

    def _get_state(self):
        left = self.car_lane == 0
        right = self.car_lane == 2
        car_pos = [int(self.car_lane == i) for i in range(3)]
        obstacles = [0, 0, 0]
        closest_dist = 1.0
        for obs in self.obstacles:
            if obs['y'] + obs['size'] > self.car_y - 300:
                lane, dist = obs['lane'], (obs['y'] + obs['size'] - self.car_y) / self.height
                if dist < closest_dist:
                    closest_dist = dist
                    obstacles[lane] = 1
        return np.array(car_pos + obstacles + [closest_dist, self.road_pos / self.height], dtype=np.float32)

    def _is_collision(self):
        car_rect = pygame.Rect(self.car_x - self.car_size//2, self.car_y - self.car_size, self.car_size, self.car_size)
        for obs in self.obstacles:
            obs_rect = pygame.Rect(obs['lane'] * self.lane_width + (self.lane_width - self.obstacle_size)//2,
                                   obs['y'], self.obstacle_size, self.obstacle_size)
            if car_rect.colliderect(obs_rect): return True
        return False

    def _check_obstacle_passed(self):
        for obs in self.obstacles:
            if obs['id'] not in self.passed_obstacles and obs['y'] + self.obstacle_size < self.car_y:
                self.score += 1
                self.passed_obstacles.add(obs['id'])

    def step(self, action):
        self.steps += 1
        if action == 1 and self.car_lane > 0: self.car_lane -= 1
        elif action == 2 and self.car_lane < self.num_lanes - 1: self.car_lane += 1
        self.car_x = self.lane_width // 2 + self.car_lane * self.lane_width
        self.road_pos = (self.road_pos + self.movement_speed) % self.height
        for obs in self.obstacles: obs['y'] += self.movement_speed
        self._check_obstacle_passed()
        for obs in self.obstacles:
            if obs['y'] > self.height:
                obs['y'] = min(obs['y'] for obs in self.obstacles) - self.obstacle_gap
                obs['lane'] = (self.obstacle_pattern[self.pattern_index] - 1) % self.num_lanes
                self.passed_obstacles.discard(obs['id'])
                self.pattern_index = (self.pattern_index + 1) % len(self.obstacle_pattern)
        if self._is_collision(): self.game_over = True; return self._get_state(), -10, True, {'score': self.score}
        return self._get_state(), 1.0, False, {'score': self.score}

    def render(self, mode='rgb_array'):
        self.surface.fill((50, 50, 50))
        for i in range(-1, 2):
            pygame.draw.rect(self.surface, (100, 100, 100), (0, self.road_pos + i * self.height, self.width, self.height))
        for lane in range(1, self.num_lanes):
            x = lane * self.lane_width
            for y in range(0, self.height, 70):
                pygame.draw.line(self.surface, (255, 255, 255),
                                 (x, (y + self.road_pos) % self.height),
                                 (x, (y + 40 + self.road_pos) % self.height), 6)
        for obs in self.obstacles:
            pygame.draw.rect(self.surface, (255, 0, 0),
                             (obs['lane'] * self.lane_width + (self.lane_width - self.obstacle_size)//2,
                              obs['y'], self.obstacle_size, self.obstacle_size))
        pygame.draw.rect(self.surface, (0, 100, 255),
                         (self.car_x - self.car_size//2, self.car_y - self.car_size, self.car_size, self.car_size))
        return np.transpose(np.array(pygame.surfarray.pixels3d(self.surface)), axes=(1, 0, 2))

# --- [3] DQN ---
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(input_size, 128), nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
    def forward(self, x):
        x = torch.relu(self.fc1(x)); x = torch.relu(self.fc2(x)); return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.model = DQN(state_size, action_size)
        self.memory, self.gamma = deque(maxlen=100000), 0.95
        self.epsilon, self.epsilon_min, self.epsilon_decay = 1.0, 0.01, 0.995
        self.batch_size, self.learning_rate = 64, 0.001
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, s, a, r, s2, d): self.memory.append((s, a, r, s2, d))
    def act(self, s):
        if np.random.rand() <= self.epsilon: return random.randint(0, 2)
        with torch.no_grad(): return torch.argmax(self.model(torch.FloatTensor(s))).item()

    def replay(self):
        if len(self.memory) < self.batch_size: return
        minibatch = random.sample(self.memory, self.batch_size)
        s = torch.FloatTensor([x[0] for x in minibatch])
        a = torch.LongTensor([x[1] for x in minibatch]).unsqueeze(1)
        r = torch.FloatTensor([x[2] for x in minibatch])
        s2 = torch.FloatTensor([x[3] for x in minibatch])
        d = torch.FloatTensor([x[4] for x in minibatch])
        current_q = self.model(s).gather(1, a).squeeze()
        with torch.no_grad():
            max_q = self.model(s2).max(1)[0]
            target_q = r + (1 - d) * self.gamma * max_q
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay
    def save(self, path): torch.save(self.model.state_dict(), path)
    def load(self, path): self.model.load_state_dict(torch.load(path)); self.model.eval()

# --- [4] Training ---
def train_agent(episodes=200):
    env = CarGame()
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    scores = []
    for e in range(episodes):
        state = env.reset(); done = False; total_reward = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay(); state = next_state; total_reward += reward
        scores.append(info['score'])
        print(f"Ep {e+1}/{episodes}, Score: {info['score']}, Epsilon: {agent.epsilon:.2f}")
    agent.save("car_dqn.pth")
    return agent

# --- [5] Save Dataset Episodes (Consolidated Tensors Version) ---
def map_action_string(a): return ["forward", "left", "right"][a]

def save_car_episodes(agent, episodes=10, out_dir="./dataset"):
    os.makedirs(out_dir, exist_ok=True)
    env = CarGame()
    
    for ep in range(1, episodes + 1):
        # Initialize lists to store episode data
        previous_frames = []
        actions = []
        target_frames = []
        
        # Initial state
        state = env.reset()
        raw_frame = env.render()
        F0 = normalize_frame(raw_frame)
        
        # Add initial transition
        previous_frames.append(STANDARD_GREY_IMAGE_TENSOR)
        actions.append(GAME_NAME)
        target_frames.append(F0)
        
        current_frame = F0
        done = False
        
        while not done:
            # Get action and step environment
            action = agent.act(state)
            state, _, done, info = env.step(action)
            raw_frame = env.render()
            next_frame = normalize_frame(raw_frame)
            
            # Store transition
            previous_frames.append(current_frame)
            actions.append(map_action_string(action))
            target_frames.append(next_frame)
            
            current_frame = next_frame
        
        # Add final transition
        previous_frames.append(current_frame)
        actions.append("<exit>")
        target_frames.append(STANDARD_GREY_IMAGE_TENSOR)
        
        # Convert to tensors and save
        episode_dict = {
            'previous_frames': torch.stack(previous_frames),
            'actions': actions,
            'target_frames': torch.stack(target_frames),
            'score': info['score']
        }
        
        torch.save(episode_dict, os.path.join(out_dir, f"{GAME_NAME}_epi{ep}.pth"))
        print(f"Saved episode {ep} with score {info['score']}")

# --- [6] Visualize Dataset (Updated for Consolidated Tensors) ---
def visualize_episode(episode_dict):
    prev_frames = episode_dict['previous_frames']
    target_frames = episode_dict['target_frames']
    actions = episode_dict['actions']
    
    for i in range(len(actions)):
        pf = denormalize_frame(prev_frames[i])
        tf = denormalize_frame(target_frames[i])
        a = actions[i]
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(pf); ax[0].set_title("Prev"); ax[0].axis("off")
        ax[1].imshow(tf); ax[1].set_title(f"Action: {a}"); ax[1].axis("off")
        plt.show()

def visualize_car_dataset(folder="./dataset"):
    for file in sorted([f for f in os.listdir(folder) if f.startswith("car_epi")]):
        path = os.path.join(folder, file)
        ep_data = torch.load(path)
        print(f"Showing {file} with score {ep_data['score']} and {len(ep_data['actions'])} transitions")
        visualize_episode(ep_data)

# --- [7] Run everything ---
if __name__ == "__main__":
    agent = train_agent(episodes=200)
    save_car_episodes(agent, episodes=50)
