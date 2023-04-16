import os
import gym
import cv2
import random
import openai
import imageio
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

    
def draw_arrow(frame, cart_position, action):
    # Map cart_position from (-2.4, 2.4) to (50, 550) to match frame dimensions
    cart_x = int(50 + (cart_position + 2.4) * (500 / 4.8))
    arrow_start = (cart_x, 300)

    if action == 0:  # Left
        arrow_end = (cart_x - 30, 300)
    else:  # Right
        arrow_end = (cart_x + 30, 300)

    # Draw arrow
    cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 255, 0), 2, tipLength=0.5)


def send_message_to_gpt(prompt_message, messages, model_name):
    """Send a message to a GPT model and return its response."""

    response = openai.ChatCompletion.create(
        model=model_name,
        messages = messages + [prompt_message],
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.01,
    )

    answer_message = dict(response['choices'][0]['message'])
    return answer_message


def gpt4_cartpole_action(observation, messages):
    """Request a GPT for the next action given the observation."""

    prompt = f"Given the Cartpole observations {observation} and the strategy provided, should the cart move [left] or [right]? Give me your chain of thoughts. Be concise!"
    prompt_message = {"role": "user", "content": prompt}
    answer_message = send_message_to_gpt(prompt_message, messages, model_name='gpt-3.5-turbo')
    
    prompt = f"Given these thoughts: '''{answer_message['content']}''', how should we now push the cart: [left] or [right]? ([left]/[right])"
    prompt_message = {"role": "user", "content": prompt}
    answer_message = send_message_to_gpt(prompt_message, messages, model_name='gpt-3.5-turbo')
    answer = answer_message['content']
    print(answer)
    
    if '[left]' in answer:
        return 0
    elif '[right]' in answer:
        return 1
    else:
        print('Wrong answer, so we just flip a coin.')
        return 1*(random.random()>0.5) #raise ValueError("Unexpected response from GPT")


def main():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    total_rewards = []

    # Create a list to store the frames
    frames = []
    
    system_message = {
    "role": "system",
    "content": "You are an AI model assisting in solving the CartPole problem. Your goal is to help balance a pole on a moving cart by providing the optimal action at each step. The cart can move in only two directions: left or right. When responding, please use the terms '[left]' or '[right]' to indicate your suggested action, making it easier for the cart to follow your advice."
    }
    
    obs_desc_text = Path('action_and_observation_description.txt').read_text()
    obs_description_message = {
        "role": "user",
        "content": obs_desc_text
    }
    
    messages = [system_message, obs_description_message]

    for episode in range(1):
        observation = env.reset()[0]
        episode_reward = 0
        done = False

        while not done:
            # Render the environment and capture the frame
            frame = env.render()
            
            action = gpt4_cartpole_action(observation, messages)
            draw_arrow(frame, observation[0], action)  # Add this line to draw an arrow
            
            frames.append(frame)
            observation, reward, done, *_ = env.step(action)
            episode_reward += reward

            # Save the frames as a GIF
            imageio.mimsave("cartpole_episode.gif", frames + [np.ones_like(frame)*200]*12, fps=24)

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: {episode_reward}")

    env.close()
    print(f"Average reward: {np.mean(total_rewards)}")

    frames += [np.ones_like(frame)*200]*12

    # Save the frames as a GIF
    imageio.mimsave("cartpole_episode.gif", frames, fps=24)

if __name__=="__main__":
    main()
