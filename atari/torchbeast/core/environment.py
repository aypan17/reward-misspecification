# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The environment class for MonoBeast."""

import torch


def _format_frame(frame):
    frame = torch.from_numpy(frame)
    return frame.view((1, 1) + frame.shape)  # (...) -> (T,B,...).


class Environment:
    def __init__(self, gym_env):
        self.gym_env = gym_env
        self.episode_return = None
        self.episode_true_return = None
        self.episode_true_move = None
        self.episode_step = None

    def initial(self):
        initial_reward = torch.zeros(1, 1)
        initial_true_reward = torch.zeros(1, 1)
        # This supports only single-tensor actions ATM.
        initial_last_action = torch.zeros(1, 1, dtype=torch.int64)
        self.episode_return = torch.zeros(1, 1)
        self.episode_true_return = torch.zeros(1, 1)
        self.episode_true_move = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.ones(1, 1, dtype=torch.uint8)
        initial_frame = _format_frame(self.gym_env.reset())
        return dict(
            frame=initial_frame,
            reward=initial_reward,
            true_reward=initial_true_reward,
            done=initial_done,
            episode_return=self.episode_return,
            episode_true_return=self.episode_true_return,
            episode_true_move=self.episode_true_move,
            episode_step=self.episode_step,
            last_action=initial_last_action,
        )

    def step(self, action):
        frame, reward, done, unused_info = self.gym_env.step(action.item())
        true_reward = self.gym_env.true_reward
        true_move = self.gym_env.true_move
        self.episode_step += 1
        self.episode_return += reward
        self.episode_true_return += true_reward
        self.episode_true_move += true_move
        episode_step = self.episode_step
        episode_return = self.episode_return
        episode_true_return = self.episode_true_return
        episode_true_move = self.episode_true_move
        if done:
            # _ = self.gym_env.reset()
            # for i in range(10):
            #     frame, _, _, _ = self.gym_env.step(2)
            #     frame, _, _, _ = self.gym_env.step(17)
            frame = self.gym_env.reset()
            self.episode_return = torch.zeros(1, 1)
            self.episode_true_return = torch.zeros(1, 1)
            self.episode_true_move = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)

        frame = _format_frame(frame)
        reward = torch.tensor(reward).view(1, 1)
        true_reward = torch.tensor(true_reward).view(1, 1)
        true_move = torch.tensor(true_move).view(1, 1)
        done = torch.tensor(done).view(1, 1)

        return dict(
            frame=frame,
            reward=reward,
            true_reward=true_reward,
            done=done,
            episode_return=episode_return,
            episode_true_return=episode_true_return,
            episode_true_move=episode_true_move,
            episode_step=episode_step,
            last_action=action,
        )

    @property
    def ram(self):
        return self.gym_env.unwrapped._get_ram()

    def close(self):
        self.gym_env.close()
