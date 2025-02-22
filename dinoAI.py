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
# 2.  Build the Environment
# 2.1 Create Environment
class WebGame(Env):
    def __init__(self):
        super().__init__()
        # Set up Spaces
        self.observation_space = Box(low=0, high=255, shape=(1,83,100), dtype=np.uint8)
        self.action_space = Discrete(3)
        # Capture game frames
        self.game_location = {'top': 300, 'left': 0, 'width': 600, 'height': 500}
        self.done_location = {'top': 405, 'left': 640, 'width': 660, 'height': 70}
    
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
        raw = np.array(self.cap.grab(self.game_location)[:,:,3].astype(np.uint8))
        # Preprocess the data -> greyscale and resize
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (100, 83))
        channel = np.reshape(resized, (1, 83, 100))
        return channel
    
    def get_done(self):
        done_cap = np.array(self.cap.grab(self.done_location)[:,:,3].astype(np.uint8))
        done_string = ['GAME', 'GAHE']
        done = False
        res = pytesseract.image_to_string(done_cap)[:4]
        if res in done_string:
            done = True
        return done, done_cap
    

# 2.2 Test the Environment

# 3. Train Model
# 3.1 Create Callback
# 3.2 Build DQN Model and Train

# 4. Test out Model