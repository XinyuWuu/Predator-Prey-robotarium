import itertools
import time
from copy import deepcopy

# import gym
import numpy as np
import torch

# from spinup.utils.logx import EpochLogger
from torch.optim import Adam

# import spinup.algos.pytorch.sac.core as core
import core
import simulator


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(
            size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(
            size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(
            size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size, device):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k, v in batch.items()}


def sac(env=simulator.Simulator(2, show_figure=False),
        test_env=simulator.Simulator(2, show_figure=True),
        obs_dim=8,
        act_dim=2,
        act_limit=np.array([0.2, 3.63]),
        hidden_sizes=(256, 256, 256),
        activation=torch.nn.ReLU,
        device="cuda",
        seed=0,
        steps_per_epoch=6000,
        epochs=1000,
        replay_size=int(1e6),
        gamma=0.99,
        polyak=0.995,
        lr=1e-3,
        alpha=0.005,
        batch_size=256,
        random_steps=10000,
        update_after=1000,
        update_every=50,
        num_test_episodes=1,
        max_ep_len=3000):

    torch.manual_seed(seed)
    np.random.seed(seed)

    ac = core.MLPActorCritic(
        obs_dim, act_dim, act_limit, hidden_sizes, device, activation)
    ac_targ = deepcopy(ac)
    ac.to(device)
    ac_targ.to(device)

    log_alpha = torch.tensor(np.log(alpha), dtype=torch.float,
                             device=device, requires_grad=True)
    alpha = np.exp(log_alpha.cpu().detach().numpy())
    alpha_optim = torch.optim.Adam([log_alpha], lr=lr)

    for p in ac_targ.parameters():
        p.requires_grad = False

    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    def compute_loss_q(data):

        q1 = ac.q1(data["obs"], data['act'])
        q2 = ac.q2(data["obs"], data['act'])

        with torch.no_grad():
            a2, logp_a2 = ac.pi(data["obs2"])

            q1_pi_targ = ac_targ.q1(data["obs2"], a2)
            q2_pi_targ = ac_targ.q2(data["obs2"], a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = data['rew'] + gamma * \
                (1 - data['done']) * (q_pi_targ - alpha * logp_a2)

        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q

    # Set up function for computing SAC pi loss

    def compute_loss_pi(data):
        pi, logp_pi = ac.pi(data['obs'])
        q1_pi = ac.q1(data['obs'], pi)
        q2_pi = ac.q2(data['obs'], pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        return loss_pi

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        with torch.no_grad():
            a, logp = ac.pi(data['obs'])
        loss_alpha = -(log_alpha.exp() * (logp.detach() - act_dim / 2)).mean()
        alpha_optim.zero_grad()
        loss_alpha.backward()
        alpha_optim.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32).to(device),
                      deterministic)

    def test_agent():
        total_ret, total_len, succ_rate = 0, 0, 0
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len, succ, = test_env.reset(
                np.array([[2], [2], [0]])), False, 0, 0, False
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                a = get_action(o, True)
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

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs

    o, ep_ret, ep_len = env.reset((np.random.rand(3, 2) * 2 - 1) * 3), 0, 0

    start_time = time.time()
    max_ret, max_ret_time, max_ret_rel_time =  \
        -1e6, time.time(), (time.time() - start_time) / 3600
    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        if t > random_steps:
            a = get_action(o)
        else:
            a = (np.random.rand(act_dim) * 2 - 1) * act_limit

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            # logger.store(EpRet=ep_ret, EpLen=ep_len)
            print(
                f"t: {t}, ep_ret: {ep_ret:.2f}, ep_len: {ep_len}, last_r: {r:.2f}")
            o, ep_ret, ep_len = env.reset(
                (np.random.rand(3, 2) * 2 - 1) * 3), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size, device)
                alpha = np.exp(log_alpha.cpu().detach().numpy())
                update(data=batch)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            # Save model
            # if (epoch % save_freq == 0) or (epoch == epochs):
            # logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            print("!!!!!test!!!!!")
            ave_ret, ave_len, succ_rate = test_agent()
            print(
                f"test result: ret: {ave_ret:.2f}, len: {ave_len:.2f}, time: {((time.time() - start_time) / 3600):.2f}, alpha: {alpha:.2f}")
            if ave_ret > max_ret or succ_rate > 0.9:
                max_ret = ave_ret
                max_ret_time = time.time()
                max_ret_rel_time = (time.time() - start_time) / 3600
                save_prefix = f"{time.ctime(max_ret_time)}_{max_ret_rel_time:.2f}h_{max_ret:.2f}_{ave_len:.0f}"
                torch.save(ac.state_dict(),
                           f'module_saves/rew/{save_prefix}_ac.ptd')
                torch.save(ac_targ.state_dict(),
                           f'module_saves/rew/{save_prefix}_ac_targ.ptd')


torch.set_num_threads(torch.get_num_threads())
sac()

# env=simulator.Simulator(2, show_figure=False),
# test_env=simulator.Simulator(2, show_figure=True),
# obs_dim=8,
# act_dim=2,
# act_limit=np.array([0.2, 3.63]),
# hidden_sizes=(256, 256, 256),
# activation=torch.nn.ReLU,
# device="cuda",
# seed=0,
# steps_per_epoch=6000,
# epochs=1000,
# replay_size=int(1e6),
# gamma=0.99,
# polyak=0.995,
# lr=1e-3,
# alpha=0.005,
# batch_size=256,
# random_steps=10000,
# update_after=1000,
# update_every=50,
# num_test_episodes=1,
# max_ep_len=3000
