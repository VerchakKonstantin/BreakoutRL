import gym
import numpy as np
import torchvision.transforms as transforms


# Class to convert images to grayscale and crop
class Transforms:
    def to_gray(frame1, frame2=None):
        gray_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.CenterCrop((175, 150)),
            transforms.Resize((84, 84)),
            transforms.ToTensor()
        ])

        # Subtract one frame from the other to get sense of ball and paddle direction
        if frame2 is not None:
            new_frame = gray_transform(frame2) - 0.4 * gray_transform(frame1)
        else:
            new_frame = gray_transform(frame1)

        return new_frame.numpy()


# Initializes an openai gym environment
def init_gym_env(env_path):
    env = gym.make(env_path)

    state_space = env.reset()[0].shape
    # print(state_space)
    # state_space = (state_space[2], state_space[0], state_space[1])
    state_raw = np.zeros(state_space, dtype=np.uint8)
    print(state_raw.shape)
    processed_state = Transforms.to_gray(state_raw)
    state_space = processed_state.shape
    action_space = env.action_space.n
    return env, state_space, action_space
