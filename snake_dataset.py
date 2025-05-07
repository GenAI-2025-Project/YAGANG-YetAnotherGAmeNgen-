import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import pygame
import gym
from gym import spaces
from IPython.display import display, clear_output
import time
import os
import cv2
import numpy as np
import pandas as pd  # (no longer needed for metadata saving, kept here for consistency)
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import matplotlib.pyplot as plt
import pygame
import gym
from gym import spaces
from collections import deque
import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt

# 4. Training Function
def train_agent(episodes=500, render_every=50):
    env = SnakeGame(width=400, height=400, block_size=50)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    scores = []

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            if e % render_every == 0:
                env.render()

            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                scores.append(info['score'])
                print(f"Episode: {e}/{episodes}, Score: {info['score']}, Epsilon: {agent.epsilon:.2f}")
                break

            agent.replay()

    # Plot training
    plt.plot(scores)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.show()

    # Save model
    agent.save('./models/snake_dqn.pth')
    env.close()

    return agent, scores


# -----------------------------
# Global Variables (as defined in the PDF)
# -----------------------------
GAME_VOCABULARY = [ "snake_n_food"]
FRAME_SIZE = (512, 512, 3)  # (Height, Width, Channels)
NORMALIZATION_METHOD = "scaling_-1_1"
TORCH_DATA_TYPE = torch.float32
N_EPISODES_PER_GAME = 50  # or whatever target you wish

# Create STANDARD_GREY_IMAGE_TENSOR:
# Create an image with uniform RGB (128, 128, 128) of size FRAME_SIZE, then normalize.
def create_standard_grey_tensor():
    grey_img = np.full(FRAME_SIZE, 128, dtype=np.uint8)
    # Convert to float and normalize to [-1, 1]
    grey_img = grey_img.astype(np.float32) / 127.5 - 1.0
    # Convert to torch tensor with proper dtype
    return torch.tensor(grey_img, dtype=TORCH_DATA_TYPE)

STANDARD_GREY_IMAGE_TENSOR = create_standard_grey_tensor()

EPISODE_DEFINITIONS = {
    "snake_n_food": "Episode ends when the snake collides with itself or a wall."
}
ACTION_FORMATS = {
    "snake_n_food": "Single word string: 'up', 'down', 'left', 'right'."
}

# -----------------------------
# Your Snake Game Environment (unchanged)
# -----------------------------
class SnakeGame(gym.Env):
    def __init__(self, width=400, height=400, block_size=50, speed=10):
        super(SnakeGame, self).__init__()
        self.width = width
        self.height = height
        self.block_size = block_size
        self.speed = speed

        # Gym spaces
        self.action_space = spaces.Discrete(3)  # 0=straight, 1=right, 2=left
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)

        # Fixed food positions in a sequence
        self.fixed_food_positions = [
            [50, 50], [350, 250], [150, 50], [250, 50],
            [250, 350], [350, 50], [150, 350], [350, 350],
            [50, 350], [350, 150],
        ]
        self.current_food_index = 0
        self.starting_food_index = 0  # Will be randomized each episode

        # Pygame setup (headless)
        pygame.init()
        self.surface = pygame.Surface((width, height))
        self.reset()

    def reset(self):
        self.starting_food_index = random.randint(0, len(self.fixed_food_positions) - 1)
        self.current_food_index = self.starting_food_index

        # Initialize snake (three segments)
        x = self.width // 2
        y = self.height // 2
        self.snake = deque([[x, y]])
        self.snake.append([x - self.block_size, y])
        self.snake.append([x - 2 * self.block_size, y])

        self.food_pos = self.fixed_food_positions[self.current_food_index].copy()
        self.current_food_index = (self.current_food_index + 1) % len(self.fixed_food_positions)

        self.direction = 'RIGHT'
        self.next_direction = 'RIGHT'
        self.score = 0
        self.steps = 0
        self.game_over = False

        return self._get_state()

    def _place_food(self):
        self.food_pos = self.fixed_food_positions[self.current_food_index].copy()
        self.current_food_index = (self.current_food_index + 1) % len(self.fixed_food_positions)

    def _get_state(self):
        head = self.snake[0]
        danger_straight = danger_right = danger_left = 0

        if self.direction == 'LEFT':
            danger_straight = self._is_collision([head[0] - self.block_size, head[1]])
            danger_right = self._is_collision([head[0], head[1] - self.block_size])
            danger_left = self._is_collision([head[0], head[1] + self.block_size])
        elif self.direction == 'RIGHT':
            danger_straight = self._is_collision([head[0] + self.block_size, head[1]])
            danger_right = self._is_collision([head[0], head[1] + self.block_size])
            danger_left = self._is_collision([head[0], head[1] - self.block_size])
        elif self.direction == 'UP':
            danger_straight = self._is_collision([head[0], head[1] - self.block_size])
            danger_right = self._is_collision([head[0] + self.block_size, head[1]])
            danger_left = self._is_collision([head[0] - self.block_size, head[1]])
        elif self.direction == 'DOWN':
            danger_straight = self._is_collision([head[0], head[1] + self.block_size])
            danger_right = self._is_collision([head[0] - self.block_size, head[1]])
            danger_left = self._is_collision([head[0] + self.block_size, head[1]])

        # Food location relative to snake head
        food_left = food_right = food_up = food_down = 0
        if self.food_pos[0] < head[0]: food_left = 1
        elif self.food_pos[0] > head[0]: food_right = 1
        if self.food_pos[1] < head[1]: food_up = 1
        elif self.food_pos[1] > head[1]: food_down = 1

        return np.array([
            danger_straight, danger_right, danger_left,
            1 if self.direction == 'LEFT' else 0,
            1 if self.direction == 'RIGHT' else 0,
            1 if self.direction == 'UP' else 0,
            1 if self.direction == 'DOWN' else 0,
            food_left, food_right, food_up, food_down
        ], dtype=np.float32)

    def _is_collision(self, point=None):
        if point is None:
            point = self.snake[0]
        if (point[0] >= self.width or point[0] < 0 or
            point[1] >= self.height or point[1] < 0):
            return True
        if point in list(self.snake)[1:]:
            return True
        return False

    def step(self, action):
        self.steps += 1
        # Update direction based on relative action
        if action == 1:  # Right turn
            if self.direction == 'UP': self.next_direction = 'RIGHT'
            elif self.direction == 'RIGHT': self.next_direction = 'DOWN'
            elif self.direction == 'DOWN': self.next_direction = 'LEFT'
            elif self.direction == 'LEFT': self.next_direction = 'UP'
        elif action == 2:  # Left turn
            if self.direction == 'UP': self.next_direction = 'LEFT'
            elif self.direction == 'LEFT': self.next_direction = 'DOWN'
            elif self.direction == 'DOWN': self.next_direction = 'RIGHT'
            elif self.direction == 'RIGHT': self.next_direction = 'UP'
        # For action == 0, keep current direction
        self.direction = self.next_direction
        head = self.snake[0].copy()

        if self.direction == 'LEFT':
            head[0] -= self.block_size
        elif self.direction == 'RIGHT':
            head[0] += self.block_size
        elif self.direction == 'UP':
            head[1] -= self.block_size
        elif self.direction == 'DOWN':
            head[1] += self.block_size

        # Check collision
        if self._is_collision(head):
            self.game_over = True
            return self._get_state(), -10, True, {'score': self.score}

        self.snake.appendleft(head)
        if head == self.food_pos:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
            reward = -0.1

        return self._get_state(), reward, self.game_over, {'score': self.score}

    def render(self, mode='human'):
        self.surface.fill((0, 0, 0))  # Black background
        
        # Draw snake with enhanced head and eyes
        snake_length = len(self.snake)
        for i, segment in enumerate(self.snake):
            rect = pygame.Rect(segment[0], segment[1], self.block_size, self.block_size)
            
            if i == 0:  # Head - blue with detailed eyes
                # Draw head
                pygame.draw.rect(self.surface, (0, 100, 255), rect)  # Bright blue head
                
                # Eye parameters
                eye_size = self.block_size // 5
                pupil_size = eye_size // 2
                eye_offset = self.block_size // 4
                
                # Draw eyes based on direction
                if self.direction == 'RIGHT':
                    # Right eye (top right)
                    pygame.draw.rect(self.surface, (255, 255, 255),  # White eye
                        (segment[0] + self.block_size - eye_offset - eye_size,
                         segment[1] + eye_offset,
                         eye_size, eye_size))
                    pygame.draw.rect(self.surface, (0, 0, 0),  # Black pupil
                        (segment[0] + self.block_size - eye_offset - pupil_size,
                         segment[1] + eye_offset + (eye_size - pupil_size)//2,
                         pupil_size, pupil_size))
                    
                    # Left eye (bottom right)
                    pygame.draw.rect(self.surface, (255, 255, 255),
                        (segment[0] + self.block_size - eye_offset - eye_size,
                         segment[1] + self.block_size - eye_offset - eye_size,
                         eye_size, eye_size))
                    pygame.draw.rect(self.surface, (0, 0, 0),
                        (segment[0] + self.block_size - eye_offset - pupil_size,
                         segment[1] + self.block_size - eye_offset - pupil_size - (eye_size - pupil_size)//2,
                         pupil_size, pupil_size))
                        
                elif self.direction == 'LEFT':
                    # Right eye (top left)
                    pygame.draw.rect(self.surface, (255, 255, 255),
                        (segment[0] + eye_offset,
                         segment[1] + eye_offset,
                         eye_size, eye_size))
                    pygame.draw.rect(self.surface, (0, 0, 0),
                        (segment[0] + eye_offset + (eye_size - pupil_size)//2,
                         segment[1] + eye_offset + (eye_size - pupil_size)//2,
                         pupil_size, pupil_size))
                    
                    # Left eye (bottom left)
                    pygame.draw.rect(self.surface, (255, 255, 255),
                        (segment[0] + eye_offset,
                         segment[1] + self.block_size - eye_offset - eye_size,
                         eye_size, eye_size))
                    pygame.draw.rect(self.surface, (0, 0, 0),
                        (segment[0] + eye_offset + (eye_size - pupil_size)//2,
                         segment[1] + self.block_size - eye_offset - pupil_size - (eye_size - pupil_size)//2,
                         pupil_size, pupil_size))
                        
                elif self.direction == 'UP':
                    # Right eye (top right)
                    pygame.draw.rect(self.surface, (255, 255, 255),
                        (segment[0] + self.block_size - eye_offset - eye_size,
                         segment[1] + eye_offset,
                         eye_size, eye_size))
                    pygame.draw.rect(self.surface, (0, 0, 0),
                        (segment[0] + self.block_size - eye_offset - pupil_size - (eye_size - pupil_size)//2,
                         segment[1] + eye_offset + (eye_size - pupil_size)//2,
                         pupil_size, pupil_size))
                    
                    # Left eye (top left)
                    pygame.draw.rect(self.surface, (255, 255, 255),
                        (segment[0] + eye_offset,
                         segment[1] + eye_offset,
                         eye_size, eye_size))
                    pygame.draw.rect(self.surface, (0, 0, 0),
                        (segment[0] + eye_offset + (eye_size - pupil_size)//2,
                         segment[1] + eye_offset + (eye_size - pupil_size)//2,
                         pupil_size, pupil_size))
                        
                elif self.direction == 'DOWN':
                    # Right eye (bottom right)
                    pygame.draw.rect(self.surface, (255, 255, 255),
                        (segment[0] + self.block_size - eye_offset - eye_size,
                         segment[1] + self.block_size - eye_offset - eye_size,
                         eye_size, eye_size))
                    pygame.draw.rect(self.surface, (0, 0, 0),
                        (segment[0] + self.block_size - eye_offset - pupil_size - (eye_size - pupil_size)//2,
                         segment[1] + self.block_size - eye_offset - pupil_size - (eye_size - pupil_size)//2,
                         pupil_size, pupil_size))
                    
                    # Left eye (bottom left)
                    pygame.draw.rect(self.surface, (255, 255, 255),
                        (segment[0] + eye_offset,
                         segment[1] + self.block_size - eye_offset - eye_size,
                         eye_size, eye_size))
                    pygame.draw.rect(self.surface, (0, 0, 0),
                        (segment[0] + eye_offset + (eye_size - pupil_size)//2,
                         segment[1] + self.block_size - eye_offset - pupil_size - (eye_size - pupil_size)//2,
                         pupil_size, pupil_size))
            
            else:  # Body - gradient green with borders
                color_ratio = i / snake_length
                green_value = 150 - int(100 * color_ratio)
                border_color = (0, green_value + 30, 0)
                inner_color = (0, green_value, 0)
                
                # Draw border
                border_size = max(1, self.block_size // 8)
                pygame.draw.rect(self.surface, border_color, rect)
                
                # Draw inner segment
                inner_rect = pygame.Rect(
                    segment[0] + border_size,
                    segment[1] + border_size,
                    self.block_size - 2*border_size,
                    self.block_size - 2*border_size
                )
                pygame.draw.rect(self.surface, inner_color, inner_rect)

        # Draw food (red with white border)
        food_rect = pygame.Rect(self.food_pos[0], self.food_pos[1],
                               self.block_size, self.block_size)
        pygame.draw.rect(self.surface, (255, 255, 255), food_rect)  # White border
        pygame.draw.rect(self.surface, (255, 0, 0),
                       (self.food_pos[0] + 2, self.food_pos[1] + 2,
                        self.block_size - 4, self.block_size - 4))  # Red center

        # Convert to numpy array
        img = np.transpose(
            np.array(pygame.surfarray.pixels3d(self.surface)), axes=(1, 0, 2))

        if mode == 'human':
            plt.imshow(img)
            plt.axis('off')
            plt.show(block=False)
            plt.pause(1/self.speed)
            plt.clf()

        return img

    def close(self):
        pygame.quit()
# -----------------------------
# DQN Model and Agent (unchanged)
# -----------------------------
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values).item()
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([t[0] for t in minibatch]))
        actions = torch.LongTensor(np.array([t[1] for t in minibatch]))
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch]))
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch]))
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch]))
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q = self.model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        loss = self.criterion(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def save(self, filename):
        torch.save(self.model.state_dict(), filename)
    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()

def resize_and_normalize(frame, size=FRAME_SIZE):
    # Resize to FRAME_SIZE (width, height)
    resized = cv2.resize(frame, (size[1], size[0]))  # cv2 uses (width, height)
    # Convert to float32 and normalize to [-1, 1]
    normalized = resized.astype(np.float32) / 127.5 - 1.0
    # Convert from NumPy (H, W, C) to torch tensor (H, W, C)
    tensor_frame = torch.tensor(normalized, dtype=TORCH_DATA_TYPE)
    return tensor_frame

def map_action_to_direction(env, action):
    """
    For snake_n_food, convert the relative action (0, 1, 2) into the resulting absolute direction.
    Note: the environment updates its direction inside step(). After calling step(action),
    env.direction holds the new direction, which should be one of 'up', 'down', 'left', 'right'
    in lower-case.
    """
    return env.direction.lower()

def generate_dataset(
        agent=None,
        episodes=10,
        speed=10,
        dataset_folder=os.path.expanduser("./dataset")
    ):
    """
    Generate episodes following the Data Output Schema v2.0 but with consolidated tensors.
    Each episode is saved as a dictionary containing:
    - 'previous_frames': Tensor of all previous frames [N, H, W, C]
    - 'actions': List of action strings
    - 'target_frames': Tensor of all target frames [N, H, W, C]
    """
    env = SnakeGame(width=400, height=400, block_size=50, speed=speed)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Load agent if not provided
    if agent is None:
        agent = DQNAgent(state_size, action_size)
        agent.load('./models/snake_dqn.pth')

    os.makedirs(dataset_folder, exist_ok=True)

    for epi in range(1, episodes + 1):
        # Initialize lists to store data for this episode
        previous_frames = []
        actions = []
        target_frames = []
        
        state = env.reset()

        # Render and process the first actual game frame (F0)
        raw_frame = env.render(mode='rgb_array')
        F0 = resize_and_normalize(raw_frame)

        # 1. Initial Dictionary: starting marker
        previous_frames.append(STANDARD_GREY_IMAGE_TENSOR)
        actions.append("<snake_n_food>")
        target_frames.append(F0)

        # Set current_frame as F0
        current_frame = F0
        done = False

        while not done:
            # Agent chooses an action (0: straight, 1: right, 2: left)
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            # Render next frame and process it to obtain Fi+1
            raw_next_frame = env.render(mode='rgb_array')
            next_frame = resize_and_normalize(raw_next_frame)

            # Map the action to an absolute direction string.
            action_str = map_action_to_direction(env, action)

            # Store transition data
            previous_frames.append(current_frame)
            actions.append(action_str)
            target_frames.append(next_frame)

            # Prepare for next step.
            current_frame = next_frame
            state = next_state

            if done:
                break

        # 3. Final Dictionary: ending marker
        previous_frames.append(current_frame)
        actions.append("<exit>")
        target_frames.append(STANDARD_GREY_IMAGE_TENSOR)

        # Convert lists to tensors
        episode_dict = {
            'previous_frames': torch.stack(previous_frames),
            'actions': actions,
            'target_frames': torch.stack(target_frames)
        }

        # Save the episode using torch.save
        filename = os.path.join(dataset_folder, f"snake_n_food_epi{epi}.pth")
        torch.save(episode_dict, filename)
        print(f"Episode {epi} saved: Score {info['score']}")

    env.close()
    print(f"All {episodes} episodes have been generated and saved in {dataset_folder}")



        
if __name__ == "__main__":
    print("Training the agent...")
    trained_agent, training_scores = train_agent(episodes=250)
    generate_dataset(agent=trained_agent, episodes=75, speed=10)
