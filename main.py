import os
import gym
import openai
import numpy as np
from dotenv import load_dotenv


def send_message_to_gpt(prompt):
    """Send a message to a GPT model and return its response."""
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.01,
    )

    message = response['choices'][0]['message']
    
    return message['content']


def gpt4_cartpole_action(observation):
    """Request a GPT for the next action given the observation."""
    
    prompt = f"Given the Cartpole observations {observation}, should the cart move left or right? (left/right)"
    answer = send_message_to_gpt(prompt)
    print(answer)
    
    if 'left' in answer:
        return 0
    elif 'right' in answer:
        return 1
    else:
        raise ValueError("Unexpected response from GPT-4")


def main():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    env = gym.make("CartPole-v1", render_mode="human")
    total_rewards = []

    for episode in range(1):
        observation = env.reset()
        episode_reward = 0
        done = False

        while not done:
            env.render()
            action = gpt4_cartpole_action(observation)
            observation, reward, done, *_ = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: {episode_reward}")

    env.close()
    print(f"Average reward: {np.mean(total_rewards)}")

if __name__=="__main__":
    main()
