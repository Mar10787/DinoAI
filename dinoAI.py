# Using RL to train AI to play dino on google chrome browser
# chrome://dino/

# 1. Installing and Importing Dependencies
# !pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
# !pip install stable-baselines3[extra] protobuf==3.20.*
# !pip install mss pydirectinput pytesseract

from mss import mss
import pydirectinput
import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
import time
from gym import Env
from gym.spaces import Discrete, Box
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import env_checker

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

# 2.  Build the Environment
# 2.1 Create Environment
class WebGame(Env):
    def __init__(self):
        super().__init__()
        # Set up Spaces
        self.observation_space = Box(low=0, high=255, shape=(1,83,100), dtype=np.uint8)
        self.action_space = Discrete(3)
        # Capture game frames
        self.cap = mss()
        self.game_location = {'top': 200, 'left': 1450, 'width': 400, 'height': 500}
        self.done_location = {'top': 250, 'left': 1700, 'width': 500, 'height': 70}
    
    def step(self,action):
        action_map = {0: 'space', 1: 'down', 2: 'no_op'}
        if action != 2:
            pydirectinput.press(action_map[action])
            
        done, done_cap = self.get_done()
        observation = self.get_observation()
        reward = 1
        info = {}
        return observation, reward, done, info
    
    def reset(self):
        time.sleep(1)
        pydirectinput.click(x=150,y=150)
        pydirectinput.press('space')
        return self.get_observation()
    
    def render(self):
        cv2.imshow('Game', self.current_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()
    
    def close(self):
        cv2.destroyAllWindows()
    
    def get_observation(self):
        raw = self.cap.grab(self.game_location)  # Capture the screen
        img = np.array(raw)  # Convert to a NumPy array
        img = img[:, :, :3]  # Extract RGB channels
        # Preprocess the data -> greyscale and resize
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (100,83))
        channel = np.reshape(resized, (1,83,100))
        return channel
    
    def get_done(self):
        done_cap = np.array(self.cap.grab(self.done_location))
        done_string = ['GAME', 'GAHE']
        done = False
        res = pytesseract.image_to_string(done_cap)[:4]
        if res in done_string:
            done = True
        return done, done_cap
    
    
def test_environment():
    env = WebGame()
    obs = env.get_observation()
    plt.imshow(cv2.cvtColor(obs[0], cv2.COLOR_BGR2RGB))
    plt.show()
    done, done_cap = env.get_done()
    if done:
        print('Game Over')
        plt.imshow(done_cap)
    
    for episode in range(2):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            obs, reward, done, info = env.step(env.action_space.sample())
            total_reward += reward
        print('Total Reward for episode {}: {}'.format(episode, total_reward))
 
 
# 3. Train the Model
# 3.1 Creating a Callback
class TrainAndLogCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLogCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        return True
    

# 3.2 Building Model
def model():
    env = WebGame()
    callback = TrainAndLogCallback(check_freq=1000, save_path=CHECKPOINT_DIR)
    model = DQN('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, buffer_size=1200000, learning_starts = 1000)
    model.learn(total_timesteps=1000000, callback=callback)


#test_environment() 
model()