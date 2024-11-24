import gym_cutting_stock
import gymnasium as gym
import pygame
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2210xxx.policy2210xxx import Policy2210xxx

# Khởi tạo pygame
pygame.init()

# Tạo cửa sổ đồ họa với kích thước 600x400
screen = pygame.display.set_mode((600, 400))

# Đặt tiêu đề cho cửa sổ
pygame.display.set_caption("Cutting Stock Simulation")

# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",  # Comment this line to disable rendering
)
NUM_EPISODES = 100

if __name__ == "__main__":
    # Reset the environment
    observation, info = env.reset(seed=42)

    # Test GreedyPolicy
    gd_policy = GreedyPolicy()
    ep = 0
    while ep < NUM_EPISODES:
        action = gd_policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset(seed=ep)
            print(info)
            ep += 1

    # Reset the environment
    observation, info = env.reset(seed=42)

    # Test RandomPolicy
    rd_policy = RandomPolicy()
    ep = 0
    while ep < NUM_EPISODES:
        action = rd_policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset(seed=ep)
            print(info)
            ep += 1

    # Uncomment the following code to test your policy
    # # Reset the environment
    # observation, info = env.reset(seed=42)
    # print(info)

    # policy2210xxx = Policy2210xxx()
    # for _ in range(200):
    #     action = policy2210xxx.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)
    #     print(info)

    #     if terminated or truncated:
    #         observation, info = env.reset()
    # Đặt màu nền cho cửa sổ (màu đen)
    screen.fill((0, 0, 0))

    # Cập nhật cửa sổ
    pygame.display.update()

# Thoát pygame
pygame.quit()
sys.exit()

#Ctrl c to interrupt

env.close()
