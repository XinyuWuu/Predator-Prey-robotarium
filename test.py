import numpy as np
import torch

import core
import simulator

device = "cpu"
torch.manual_seed(100)
np.random.seed(100)

# hidden = (255, 255, 255)
# model_file = "module_saves/rew/Tue Oct 25 05:12:24 2022_2.67h_2721.42_1255_ac.ptd"

# hidden = (256, 256, 256)
# model_file = "module_saves/rew/Tue Oct 25 16:07:00 2022_4.81h_2178.53_1058_ac.ptd"

hidden = (256, 256, 256)
model_file = "module_saves/rew/Tue Oct 25 17:11:29 2022_5.88h_2198.14_1149_ac.ptd"

# hidden = (256, 256, 256)
# model_file = "module_saves/rew/Tue Oct 25 18:37:07 2022_7.31h_2200.07_1156_ac.ptd"

test_env = simulator.Simulator(2, show_figure=True)
ac = core.MLPActorCritic(8, 2, np.array(
    [0.2, 3.63]), hidden, device, torch.nn.ReLU)
ac.load_state_dict(torch.load(model_file,map_location=torch.device("cuda")))


def test_agent(test_env, ac, deterministic, num_test_episodes, max_ep_len):
    total_ret, total_len, succ_rate = 0, 0, 0
    for j in range(num_test_episodes):
        o, d, ep_ret, ep_len, succ, = test_env.reset(
            np.array([[2], [2], [0]])), False, 0, 0, False
        while not(d or (ep_len == max_ep_len)):
            # Take deterministic actions at test time
            a = ac.act(torch.as_tensor(o, dtype=torch.float32).to(device),
                       deterministic)
            o, r, d, _ = test_env.step(a)
            ep_ret += r
            ep_len += 1
            total_ret += r
            total_len += 1
            if d == 1 and r > 0:
                succ = True
                succ_rate += 1
        print(
            f"test result: ret: {ep_ret:.2f}, len: {ep_len}, success: {succ}")

    return total_ret / num_test_episodes, total_len / num_test_episodes, succ_rate / num_test_episodes


test_agent(test_env, ac, True, 5, 3000)
