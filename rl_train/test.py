import sys
import os

# 1. Force Python to look in the current directory for myo_assist
sys.path.append(os.getcwd())

try:
    import gymnasium as gym
except ImportError:
    import gym

try:
    # 2. Try to import the package that registers the environments
    # In MyoAssist, this is usually 'myo_assist' or 'myo_assist_envs'
    import myo_assist 
    
    env = gym.make("myoAssistLegImitationExo-v0")
    obs = env.reset()
    
    print("-" * 30)
    print(f"SUCCESS!")
    print(f"Total Observation Size: {len(obs)}")
    print("-" * 30)
    
    # Verify the target velocity index
    if len(obs) == 44:
        print("Your index [43, 44] for Target Velocity is CORRECT.")
    else:
        print(f"WARNING: Your vector size is {len(obs)}, not 44.")
        print(f"You need to shift your 'target velocity' index to [{len(obs)-1}, {len(obs)}]")

except ModuleNotFoundError:
    print("ERROR: Still can't find 'myo_assist'.")
    print("Make sure you are running this script from the main folder of the repo.")