from simulation.gym.E import E
import torch
import torch.nn as nn

import torch
import numpy as np
from simulation.file_instance import FileInstance
from simulation.greedy import get_lots_to_dispatch_by_machine
from simulation.dispatching.dispatcher import dispatcher_map
from simulation.gym.E import E
from simulation.read import read_all
from collections import defaultdict
from simulation.randomizer import Randomizer

import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.optim import Adam
from torch.distributions import Categorical
from simulation.classes import Lot
from statistics import mean, median
from logger import Logger
import matplotlib.pyplot as plt
r = Randomizer()

class SCFabEnv:
    def __init__(self, dataset, days=1, dispatcher='fifo', seed=0, state_components=None):
        self.files = read_all('datasets/' + dataset)
        self.instance = None
        self.days = days

        self.seed_val = seed
        self.dispatcher = dispatcher_map[dispatcher]
        self.state_components = state_components or [
            E.A.L4M.S.OPERATION_TYPE.CR.MAX,
            E.A.L4M.S.OPERATION_TYPE.FREE_SINCE.MAX
        ]

        self.metrics = {
            'throughput': [],
            'tardiness': [],
            'reward': [],
        }
        self._state = None
        self.lots_done = 0
        self.logger = Logger("logs")

    def process(self):
        machines_to_check = list(self.instance.usable_machines)
        machine_lot_group_pair = []
        machine_to_remove = []
        for machine in machines_to_check:
            m, lots = get_lots_to_dispatch_by_machine(
                self.instance, machine=machine, ptuple_fcn=self.dispatcher
            )
            if lots is None:
                machine_to_remove.append(machine)
                continue
                
            if machine in machine_to_remove:
                continue
            
            actions = defaultdict(list)
            for lot in m.waiting_lots:
                actions[lot.actual_step.step_name].append(lot)

            m.actions = list(actions.values())
            for action in m.actions:
                machine_lot_group_pair.append((m, action))

        self.machine_lot_group_pair = machine_lot_group_pair[:50]
        
        for machine in machine_to_remove:
            if machine in self.instance.usable_machines:
                self.instance.usable_machines.remove(machine)
        
        self._state = None


    def get_machine_state(self, m, action):
        t = self.instance.current_time
        _state = [
            m.pms[0].timestamp - t if len(m.pms) > 0 else 999999,  # next maintenance
            m.utilized_time / m.setuped_time if m.setuped_time > 0 else 0,  # ratio of setup time / processing time
            (m.setuped_time + m.utilized_time) / t if t > 0 else 0,  # ratio of non idle time
            m.machine_class,  # type of machine
        ]
        
        if action is None:
            _state += [-1000] * len(self.state_components)
        else:
            action: list[Lot]
            free_since = [self.instance.current_time - l.free_since for l in action]
            work_rem = [len(l.remaining_steps) for l in action]
            cr = [l.cr(self.instance.current_time) for l in action]
            priority = [l.priority for l in action]
            l0 = action[0]

            action_type_state_lambdas = {
                E.A.L4M.S.OPERATION_TYPE.NO_LOTS: lambda: len(action),
                E.A.L4M.S.OPERATION_TYPE.NO_LOTS_PER_BATCH: lambda: len(action) / l0.actual_step.batch_max,
                E.A.L4M.S.OPERATION_TYPE.STEPS_LEFT.MEAN: lambda: mean(work_rem),
                E.A.L4M.S.OPERATION_TYPE.STEPS_LEFT.MEDIAN: lambda: median(work_rem),
                E.A.L4M.S.OPERATION_TYPE.STEPS_LEFT.MAX: lambda: max(work_rem),
                E.A.L4M.S.OPERATION_TYPE.STEPS_LEFT.MIN: lambda: min(work_rem),
                E.A.L4M.S.OPERATION_TYPE.FREE_SINCE.MEAN: lambda: mean(free_since),
                E.A.L4M.S.OPERATION_TYPE.FREE_SINCE.MEDIAN: lambda: median(free_since),
                E.A.L4M.S.OPERATION_TYPE.FREE_SINCE.MAX: lambda: max(free_since),
                E.A.L4M.S.OPERATION_TYPE.FREE_SINCE.MIN: lambda: min(free_since),
                E.A.L4M.S.OPERATION_TYPE.PROCESSING_TIME.AVERAGE: lambda: l0.actual_step.processing_time.avg(),
                E.A.L4M.S.OPERATION_TYPE.BATCH.MIN: lambda: l0.actual_step.batch_min,
                E.A.L4M.S.OPERATION_TYPE.BATCH.MAX: lambda: l0.actual_step.batch_max,
                E.A.L4M.S.OPERATION_TYPE.BATCH.FULLNESS: lambda: min(1, len(action) / l0.actual_step.batch_max),
                E.A.L4M.S.OPERATION_TYPE.PRIORITY.MEAN: lambda: mean(priority),
                E.A.L4M.S.OPERATION_TYPE.PRIORITY.MEDIAN: lambda: median(priority),
                E.A.L4M.S.OPERATION_TYPE.PRIORITY.MAX: lambda: max(priority),
                E.A.L4M.S.OPERATION_TYPE.PRIORITY.MIN: lambda: min(priority),
                E.A.L4M.S.OPERATION_TYPE.CR.MEAN: lambda: mean(cr),
                E.A.L4M.S.OPERATION_TYPE.CR.MEDIAN: lambda: median(cr),
                E.A.L4M.S.OPERATION_TYPE.CR.MAX: lambda: max(cr),
                E.A.L4M.S.OPERATION_TYPE.CR.MIN: lambda: min(cr),
                E.A.L4M.S.OPERATION_TYPE.SETUP.NEEDED: lambda: 0 if l0.actual_step.setup_needed == '' or l0.actual_step.setup_needed == m.current_setup else 1,
                E.A.L4M.S.OPERATION_TYPE.SETUP.MIN_RUNS_LEFT: lambda: 0 if m.min_runs_left is None else m.min_runs_left,
                E.A.L4M.S.OPERATION_TYPE.SETUP.MIN_RUNS_OK: lambda: 1 if l0.actual_step.setup_needed == '' or l0.actual_step.setup_needed == m.min_runs_setup else 0,
                E.A.L4M.S.OPERATION_TYPE.SETUP.LAST_SETUP_TIME: lambda: m.last_setup_time,
                E.A.L4M.S.MACHINE.MAINTENANCE.NEXT: lambda: 0,
                E.A.L4M.S.MACHINE.IDLE_RATIO: lambda: 1 - (
                            m.utilized_time / self.instance.current_time) if self.instance.current_time > 0 else 1,
                E.A.L4M.S.MACHINE.SETUP_PROCESSING_RATIO: lambda: (
                            m.setuped_time / m.utilized_time) if m.utilized_time > 0 else 1,
                E.A.L4M.S.MACHINE.MACHINE_CLASS: lambda: 0,
            }
            _state += [
                action_type_state_lambdas[s]()
                for s in self.state_components
            ]
        return _state

    @property
    def state(self):
        if self._state is None:
            state = []
            for m, lot_group in self.machine_lot_group_pair:
                state.append(self.get_machine_state(m, lot_group))
            state = np.array(state).astype(np.float32)
            self._state = state
        return self._state
        
    def reset(self):
        run_to = 3600 * 24 * self.days
        self.lots_done = 0
        self.instance = FileInstance(self.files, run_to, True, [])
        self.process()
        self.eid = np.random.randint(999_999_999)
        return self.state

    def step(self, action):
        if len(self.machine_lot_group_pair) > 0:
            machine, lot_group = self.machine_lot_group_pair[action]
            lot = lot_group[0]
            lots = lot_group[:min(len(lot_group), lot.actual_step.batch_max)]
            violated_minruns = machine.min_runs_left is not None and machine.min_runs_setup == lot.actual_step.setup_needed
            self.instance.dispatch(machine, lots)

        step_throughput = 0
        step_tardiness = 0

        done = self.instance.next_decision_point()
        if done or self.instance.current_time > 3600 * 24 * self.days:
            done = True

        reward = 0
        # if violated_minruns:
        #     reward += -1

        for i in range(self.lots_done, len(self.instance.done_lots)):
            lot = self.instance.done_lots[i]
            reward += 1 if lot.deadline_at >= lot.done_at else -1

        new_lots_done = self.instance.done_lots[self.lots_done:]

        for lot in new_lots_done:
            step_throughput += 1
            lateness_hours = max(0, (lot.done_at - lot.deadline_at) / 3600)
            step_tardiness += lateness_hours

        self.metrics['throughput'].append(step_throughput)
        self.metrics['tardiness'].append(step_tardiness)
        self.metrics['reward'].append(reward)

        self.lots_done = len(self.instance.done_lots)
        self.process()
        self.logger.add_to_pool(eid=self.eid, 
                                time=self.instance.current_time,
                                num_actions=len(self.machine_lot_group_pair),
                                reward=reward,
                                throughput=step_throughput,
                                tardiness=step_tardiness)
        env.logger.commit()
        return self.state, reward, done, {}
    

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

    def forward(self, x):
        return x + self.pos_embedding[:, :x.shape[1], :]

class FeatureExtractor(nn.Module):
    def __init__(self, seq_len, input_dim, num_layers=4, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.pos_encoder = LearnablePositionalEncoding(seq_len, input_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x): 
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return x

class EMB(nn.Module):
    def __init__(self, embed_size, hdim, drp=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, hdim),
            nn.Dropout(drp),
            nn.GELU(),
            nn.Linear(hdim, hdim),
            nn.Dropout(drp),
            nn.GELU(),
            nn.Linear(hdim, embed_size),
        )
    def forward(self, x):
        #N,L,E -> N,L,E
        x = self.mlp(x)
        return x.squeeze(-1)


class Actor(nn.Module):
    def __init__(self, embed_size, hdim, drp=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, hdim),
            nn.Dropout(drp),
            nn.GELU(),
            nn.Linear(hdim, hdim),
            nn.Dropout(drp),
            nn.GELU(),
            nn.Linear(hdim, 1),
        )
    def forward(self, x):
        #N,L,E -> N,L,1
        x = self.mlp(x)
        return x.squeeze(-1)
    
class critic(nn.Module):
    def __init__(self, embed_size, hdim, drp=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, hdim),
            nn.Dropout(drp),
            nn.GELU(),
            nn.Linear(hdim, hdim),
            nn.Dropout(drp),
            nn.GELU(),
            nn.Linear(hdim, 1),
        )
    def forward(self, x):
        #N,L,E -> N,1
        x = self.mlp(x.mean(1))
        return x.squeeze(-1)

class Model(nn.Module):
    def __init__(self, input_dim, embed_size, seq_len, num_enc, num_heads, hdim, drp=0.1):
        super().__init__()

        self.embedding = nn.Linear(input_dim, embed_size)
        self.fe = EMB(embed_size, hdim, drp)
        self.feature_extractor = FeatureExtractor(seq_len=seq_len, 
                                                  input_dim=embed_size, 
                                                  num_layers=num_enc, 
                                                  nhead=num_heads, 
                                                  dim_feedforward=hdim, 
                                                  dropout=drp)
        self.actor = Actor(embed_size, hdim, drp)
        self.critic = critic(embed_size, hdim, drp)
    
    def forward(self, x):
        # x: (N, L, E)

        unsqueezed = False
        if x.ndim == 2:
            x = x.unsqueeze(0)
            unsqueezed = True

        x = self.embedding(x)
        x_a = self.fe(x)
        x_c = self.feature_extractor(x_a)  # (N, L, E)
        logits, values = self.actor(x_a), self.critic(x_c)
        if unsqueezed:
            logits = logits.squeeze(0)
            values = values.squeeze(0)
        return logits, values

def collect_rollout(env, model, rollout_len=2048):
    obs_buf, action_buf, reward_buf, done_buf, logp_buf, value_buf = [], [], [], [], [], []
    obs = env.reset()
    for counter in range(rollout_len):
        store = False
        if obs.shape[0] == 0:
            action = 0
        elif obs.shape[0] == 1:
            action = 0
        else:
            with torch.no_grad():
                logits, value = model(torch.from_numpy(obs))
            probs = F.softmax(logits, dim=0)
            dist = Categorical(probs)
            action = dist.sample()
            logp_buf.append(dist.log_prob(action))
            value_buf.append(value.squeeze(-1))
            obs_buf.append(obs)
            action_buf.append(action)
            store = True

        next_obs, reward, done, _ = env.step(action if isinstance(action, int) else action.item())
        if store:
            reward_buf.append(torch.tensor(reward, dtype=torch.float32))
            done_buf.append(done)
        obs = next_obs
        if done:
            print("Episode done, resetting environment")
            break
    print("counter: ", counter, "time:", {np.round(env.instance.current_time/3600/24, 3)})

    return obs_buf, action_buf, reward_buf, done_buf, logp_buf, value_buf


def ppo_update(model, optimizer, obs_buf, action_buf, reward_buf, done_buf, logp_buf, value_buf,
               gamma=0.95, lam=0.95, clip_ratio=0.2, epochs=5, batch_size=32):

    returns = []
    advs = []
    gae = 0
    last_value = 0

    ploss, vloss = [], []
    for t in reversed(range(len(reward_buf))):
        mask = 1.0 - float(done_buf[t])
        delta = reward_buf[t] + gamma * last_value * mask - value_buf[t]
        gae = delta + gamma * lam * mask * gae
        advs.insert(0, gae)
        last_value = value_buf[t]
        returns.insert(0, gae + value_buf[t])

    advs = torch.tensor(advs, dtype=torch.float32, requires_grad=False)
    returns = torch.tensor(returns, dtype=torch.float32, requires_grad=False)

    for _ in range(epochs):
        for i in range(0, len(obs_buf), batch_size):
            var = [model(torch.from_numpy(i)) for i in obs_buf[i:i+batch_size]]
            logits, new_values = [i[0] for i in var], torch.tensor([i[1] for i in var])
            dists = [Categorical(logits=l) for l in logits]

            act_batch = action_buf[i:i+batch_size]
            old_logp_batch = logp_buf[i:i+batch_size]

            new_logp = []
            for g in range(len(act_batch)):
                dist = dists[g]
                action = act_batch[g]
                log_prob = dist.log_prob(action)
                new_logp.append(log_prob)

            ratio = [torch.exp(new_logp_i - old_logp_batch_i) for new_logp_i, old_logp_batch_i in zip(new_logp, old_logp_batch)]
            adv_batch = advs[i:i+batch_size]
            ret_batch = returns[i:i+batch_size]

            surr1 = [r*a for r, a in zip(ratio, adv_batch)]
            surr2 = [torch.clamp(r, 1.0-clip_ratio, 1.0+clip_ratio) * a for r, a in zip(ratio, adv_batch)]
            policy_loss = -sum([min(s1, s2) for s1, s2 in zip(surr1, surr2)])/len(surr1)

            value_loss = F.mse_loss(new_values.squeeze(-1), ret_batch)
            loss = policy_loss + 0.25 * value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ploss.append(policy_loss.item())
            vloss.append(value_loss.item())
    return np.mean(ploss), np.mean(vloss)

state_components = (E.A.L4M.S.OPERATION_TYPE.NO_LOTS,
                    E.A.L4M.S.OPERATION_TYPE.NO_LOTS_PER_BATCH,
                    E.A.L4M.S.OPERATION_TYPE.STEPS_LEFT.MEAN,
                    E.A.L4M.S.OPERATION_TYPE.STEPS_LEFT.MEDIAN,
                    E.A.L4M.S.OPERATION_TYPE.STEPS_LEFT.MAX,
                    E.A.L4M.S.OPERATION_TYPE.STEPS_LEFT.MIN,
                    E.A.L4M.S.OPERATION_TYPE.FREE_SINCE.MEAN,
                    E.A.L4M.S.OPERATION_TYPE.FREE_SINCE.MEDIAN,
                    E.A.L4M.S.OPERATION_TYPE.FREE_SINCE.MAX,
                    E.A.L4M.S.OPERATION_TYPE.FREE_SINCE.MIN,
                    E.A.L4M.S.OPERATION_TYPE.PROCESSING_TIME.AVERAGE,
                    E.A.L4M.S.OPERATION_TYPE.BATCH.MIN,
                    E.A.L4M.S.OPERATION_TYPE.BATCH.MAX,
                    E.A.L4M.S.OPERATION_TYPE.BATCH.FULLNESS,
                    E.A.L4M.S.OPERATION_TYPE.PRIORITY.MEAN,
                    E.A.L4M.S.OPERATION_TYPE.PRIORITY.MEDIAN,
                    E.A.L4M.S.OPERATION_TYPE.PRIORITY.MAX,
                    E.A.L4M.S.OPERATION_TYPE.PRIORITY.MIN,
                    E.A.L4M.S.OPERATION_TYPE.CR.MEAN,
                    E.A.L4M.S.OPERATION_TYPE.CR.MEDIAN,
                    E.A.L4M.S.OPERATION_TYPE.CR.MAX,
                    E.A.L4M.S.OPERATION_TYPE.CR.MIN,
                    E.A.L4M.S.OPERATION_TYPE.SETUP.NEEDED,
                    E.A.L4M.S.OPERATION_TYPE.SETUP.MIN_RUNS_LEFT,
                    E.A.L4M.S.OPERATION_TYPE.SETUP.MIN_RUNS_OK,
                    E.A.L4M.S.OPERATION_TYPE.SETUP.LAST_SETUP_TIME,
                    E.A.L4M.S.MACHINE.MAINTENANCE.NEXT,
                    E.A.L4M.S.MACHINE.IDLE_RATIO,
                    E.A.L4M.S.MACHINE.SETUP_PROCESSING_RATIO,
                    E.A.L4M.S.MACHINE.MACHINE_CLASS)

env = SCFabEnv(days=365, 
               dataset="SMT2020_HVLM", 
               dispatcher="fifo", 
               seed=42, 
               state_components=state_components)

model = Model(input_dim=34,
             embed_size=64, 
             seq_len=50, 
             num_enc=4, 
             num_heads=4,
             hdim=96)

# _ = env.reset()
# for i in range(10000000):
#     next_obs, reward, done, _ = env.step(0)
#     if done:
#         print("breaking", i)
#         break
#     try:
#         print(np.round(env.machine_lot_group_pair[0][-1][0].deadline_at/3600, 3), 
#               np.round(env.instance.current_time/3600/24, 3), 
#               len(env.instance.usable_machines), 
#               len(env.instance.done_lots))
#     except:
#         print(np.round(env.instance.current_time/3600, 3), len(env.instance.usable_machines), len(env.instance.done_lots))

optimizer = Adam(model.parameters(), lr=1e-4)
_ = env.reset()
for _ in range(10000):
    env.step(0)

print(np.round(env.machine_lot_group_pair[0][-1][0].deadline_at/3600, 3), np.round(env.instance.current_time/3600, 3), len(env.instance.usable_machines))

for iter in range(1000):
    obs_buf, action_buf, reward_buf, done_buf, logp_buf, value_buf = collect_rollout(env, model, rollout_len=100000000)
    ploss, vloss = ppo_update(model, optimizer, obs_buf, action_buf, reward_buf, done_buf, logp_buf, value_buf)
    env.logger.add_to_pool(eid=env.eid, 
                           policy_loss=ploss,
                           value_loss=vloss,
                           total_rewards=np.sum(reward_buf),
                           done_lots=len(env.instance.done_lots))
    env.logger.commit()
    print(f"[{iter}] R: {np.sum(reward_buf):.6f}, PLoss: {ploss:.6f}, VLoss: {vloss:.6f}, DoneLots: {len(env.instance.done_lots)}")
    print(f"Iteration {iter}, Throughput: {np.sum(env.metrics['throughput'])}, Tardiness: {np.sum(env.metrics['tardiness'])}, Reward: {np.sum(env.metrics['reward'])}")
    env.metrics = {'throughput': [], 'tardiness': [], 'reward': []}