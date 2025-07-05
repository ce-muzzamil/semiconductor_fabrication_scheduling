
"""
**About this Simulation:**

This simulation models a semiconductor manufacturing process, utilizing it for scheduling optimization. Our goal is to train a Reinforcement Learning (RL) agent capable of generating the most optimized
schedules to maximize throughput and minimize tardiness.

Within the simulation, resources are represented as “machines,” available as instances within the `useable_machines` list. A machine becomes available at time *t* if both of the following conditions are
met:

1.  The machine is currently free at time *t*.
2.  There are lots requiring that machine at time *t*.

The agent will have the choice to either deploy a lot (“work”) onto a machine or hold it for a period, potentially for a better overall outcome.  In this simulation, lots are collections of processing
steps. We utilize `lot.actual_step` to determine the machine required for processing a specific lot.

Our plan is to provide the agent with the following information:

*   **Lot Embeddings:** Each lot will be processed with all `lot.remaining_steps` and its features, generating a rich information tensor instead of relying solely on the `lot.actual_step`/step_t.
*   **Machine Embeddings:** We will also develop machine embeddings by analyzing the future requirements of each specific machine, creating a rich tensor representation. Once this is established, we’ll
establish a connection between these two tensors, feeding them to the agent to inform its decision-making.
"""

"""
State Representation:
the usable_machines and their waiting_lots but a single family of machine will be represented only once
the lots that are requiring a machine m of family f at time t will all be dispatched at once to a single machine instead
of using multiple instances. this is done to remian coherent with the original simulation. since this will change the 

So, at time t:
-   A set of machines
    -   Each machine has machine features, available units and, future needs based on information from the future steps of undone lots
-   A set of lots
    -   Each lot has its features, number of steps and its requirements of other machines in the future

"""

"""
There are two possibilities to solve this problem:
- The fisrt is to provide lot_machine pair and ask the agent whether to dispatch the lot to the machine or not
-The second is to dispatch what non-conflicting and ask the ganet only if there are conflicts for example
there is one machine and two lots that require it. 

Traditionally the second task is targted by the schduling and we will also try to stick with that one.
"""

    

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
        return x #N,L,E

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
        return x


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
        logits, values = self.actor(x_c), self.critic(x_c)

        if unsqueezed:
            logits = logits.squeeze(0)
            values = values.squeeze(0)

        return logits, values

def collect_rollout(env, model, rollout_len=2048):
    obs_buf, action_buf, reward_buf, done_buf, logp_buf, value_buf, info_buf = [], [], [], [], [], [], []
    obs = env.reset()
    for counter in range(rollout_len):
        store = False
        if obs.shape[0] == 0:
            action = 0
        # elif obs.shape[0] == 1:
        #     action = 0
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

        next_obs, reward, done, info = env.step(action if isinstance(action, int) else action.item())
        if store:
            reward_buf.append(torch.tensor(reward, dtype=torch.float32))
            done_buf.append(done)
            info_buf.append(info)
        obs = next_obs
        if done:
            # print("Episode done, resetting environment")
            break

    for i in range(len(info_buf)):
        for lot in info_buf[i]["done_lots"]:
            for j in range(i):
                if info_buf[j]["time"] == lot.tag:
                    if (lot.deadline_at - lot.done_at) > 0:
                        reward_buf[j] += 0
                    else:
                        reward_buf[j] += (lot.deadline_at - lot.done_at) / 3600
 
    # print("counter: ", counter, "time:", {np.round(env.instance.current_time/3600/24, 3)})
    
    return obs_buf, action_buf, reward_buf, done_buf, logp_buf, value_buf

def ppo_update(model, optimizer, obs_buf, action_buf, reward_buf, done_buf, logp_buf, value_buf,
               gamma=0.95, lam=0.95, clip_ratio=0.2, epochs=1, batch_size=32):

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

