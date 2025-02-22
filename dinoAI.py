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

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

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
        resized = cv2.resize(gray, (300, 250))
        channel = np.reshape(resized, (1, 250, 300))
        return channel
    
    def get_done(self):
        done_cap = np.array(self.cap.grab(self.done_location))
        done_string = ['GAME', 'GAHE']
        done = False
        res = pytesseract.image_to_string(done_cap)[:4]
        if res in done_string:
            done = True
        return done, done_cap
    

# 2.2 Test the Environment
env = WebGame()
"""
obs=env.get_observation()
plt.imshow(cv2.cvtColor(obs[0],cv2.COLOR_GRAY2RGB))
plt.show()

done, done_cap = env.get_done()
plt.imshow(cv2.cvtColor(done_cap, cv2.COLOR_BGR2RGB))
plt.show()
print(pytesseract.image_to_string(done_cap)[:4])
print(done)

# Testing Random Actions
"""

# 3. Train Model
# 3.1 Create Callback - Saves Model Throughout Training
# Import OS for file path management
import os
# Import BaseCallback from stable_baselines3
from stable_baselines3.common.callbacks import BaseCallback
# Check Environment
from stable_baselines3.common.env_checker import env_checker
env_checker.check_env(env)

class TrainAndLoggingCallback(BaseCallback):
    
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
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
CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'
callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)
# 3.2 Build DQN Model and Train

# 4. Test out Model