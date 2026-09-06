# ABSOLUTE PATH: src/ipomdp/envs/vector.py
# ==============================================================================
# SYNCHRONOUS VECTORIZED MULTI-AGENT ENVIRONMENT WRAPPER
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Temporal Decoupling on Episode Termination/Truncation:
#    - When an environment channel terminates or truncates at step T:
#         * True terminal observation o_T and terminal state s_T are safely stored in
#           step_infos[agent]["terminal_obs"] and step_infos[agent]["terminal_state"].
#         * The channel resets immediately to produce initial start state s_0 and start
#           observation o_0 for the subsequent episode.
#         * b_obs and b_infos["true_state"] hold clean s_0 and o_0, preventing
#           off-by-one temporal misalignments across asynchronous channel resets.
#
# 2. Asymmetric Multi-Agent Termination Guard:
#    - Evaluates joint termination across all participating agents:
#         is_term = any(res.terminations[agent].item() for agent in self.agents)
#         is_trunc = any(res.truncations[agent].item() for agent in self.agents)
#      supporting multi-agent scenarios with joint lifetime horizons.
#
# 3. Strict Batch Tensor Rank Preservation:
#    - Collates step observations, rewards, terminations, truncations, and infos across
#      num_envs channels, preserving leading batch contracts (B, ...).
# ==============================================================================

import torch
from typing import List, Callable, Dict, Tuple
from ..types import Action, Observation, State


class SyncVectorEnv:
    """
    Synchronous Vectorized Multi-Agent Environment Runner.
    Executes parallel environment instances while preserving terminal states and observations.
    """

    def __init__(self, env_fn: Callable, num_envs: int):
        """
        Instantiates parallel environment channels.

        Args:
            env_fn: Zero-argument callable returning an IPOMDPEnv instance.
            num_envs: Number of parallel environment instances (B).
        """
        self.num_envs = int(num_envs)
        self.envs = [env_fn() for _ in range(self.num_envs)]
        self.agents = list(self.envs[0].agents)

    def reset(self) -> Tuple[Dict[str, Observation], Dict[str, dict]]:
        """
        Resets all parallel environment channels and returns batched initial observations.

        Returns:
            Tuple of (batched_observations_dict, batched_infos_dict).
        """
        obs_batch, info_batch = [], []
        for env in self.envs:
            o, i = env.reset()
            obs_batch.append(o)
            info_batch.append(i)

        batched_obs = {
            agent: Observation(torch.stack([o[agent].data for o in obs_batch]))
            for agent in self.agents
        }
        batched_infos: Dict[str, dict] = {}
        for agent in self.agents:
            agent_info = {}
            for k in info_batch[0][agent].keys():
                vals = [i[agent][k] for i in info_batch]
                if k == "true_state":
                    agent_info[k] = torch.stack([
                        v.data if isinstance(v, State) else v for v in vals
                    ])
                elif isinstance(vals[0], torch.Tensor):
                    agent_info[k] = torch.stack(vals)
                else:
                    agent_info[k] = vals
            batched_infos[agent] = agent_info
        return batched_obs, batched_infos

    def step(
        self,
        batched_actions: Dict[str, Action]
    ) -> Tuple[
        Dict[str, Observation],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, dict]
    ]:
        """
        Advances all parallel environment channels by one discrete timestep.
        Preserves true ending observations o_T and terminal states s_T before channel resets.

        Args:
            batched_actions: Dictionary mapping AgentID to Action containers with batch data (B, ...).

        Returns:
            Tuple of (observations, rewards, terminations, truncations, infos).
        """
        obs_list, rews_list, terms_list, truncs_list, infos_list = [], [], [], [], []

        for i, env in enumerate(self.envs):
            single_action = {
                agent: Action(batched_actions[agent].data[i].unsqueeze(0))
                for agent in self.agents
            }
            res = env.step(single_action)

            is_term = any(res.terminations[agent].item() for agent in self.agents)
            is_trunc = any(res.truncations[agent].item() for agent in self.agents)

            step_obs = res.observations
            step_infos = res.infos

            if is_term or is_trunc:
                for agent in self.agents:
                    step_infos[agent]["terminal_obs"] = step_obs[agent].data.clone()
                    curr_ts = step_infos[agent]["true_state"]
                    step_infos[agent]["terminal_state"] = curr_ts.data.clone() if isinstance(curr_ts, State) else curr_ts.clone()

                # Reset channel for the new episode
                reset_obs, reset_infos = env.reset()

                # Align true_state to reset initial state s_0 matching reset_obs o_0
                for agent in self.agents:
                    step_infos[agent]["true_state"] = reset_infos[agent]["true_state"]

                obs_list.append(reset_obs)
                infos_list.append(step_infos)
            else:
                obs_list.append(step_obs)
                infos_list.append(step_infos)

            rews_list.append(res.rewards)
            terms_list.append(res.terminations)
            truncs_list.append(res.truncations)

        b_obs = {
            agent: Observation(torch.stack([
                o[agent].data.squeeze(0) if o[agent].data.dim() > 1 and o[agent].data.size(0) == 1 else o[agent].data
                for o in obs_list
            ]))
            for agent in self.agents
        }

        b_rews = {
            agent: torch.cat([r[agent].view(-1, 1) for r in rews_list], dim=0)
            for agent in self.agents
        }
        b_terms = {
            agent: torch.cat([t[agent].view(-1, 1) for t in terms_list], dim=0)
            for agent in self.agents
        }
        b_truncs = {
            agent: torch.cat([tr[agent].view(-1, 1) for tr in truncs_list], dim=0)
            for agent in self.agents
        }

        b_infos: Dict[str, dict] = {}
        for agent in self.agents:
            agent_dict: Dict[str, Any] = {
                "true_state": torch.stack([
                    i[agent]["true_state"].data.view(-1) if isinstance(i[agent]["true_state"], State)
                    else i[agent]["true_state"].view(-1)
                    for i in infos_list
                ]),
                "terminal_obs": torch.stack([
                    i[agent].get("terminal_obs", b_obs[agent].data[idx]).view(-1)
                    for idx, i in enumerate(infos_list)
                ]),
                "terminal_state": torch.stack([
                    i[agent]["terminal_state"].data.view(-1) if isinstance(i[agent].get("terminal_state"), State)
                    else (i[agent]["terminal_state"].view(-1) if "terminal_state" in i[agent]
                          else (i[agent]["true_state"].data.view(-1) if isinstance(i[agent]["true_state"], State)
                                else i[agent]["true_state"].view(-1)))
                    for idx, i in enumerate(infos_list)
                ])
            }
            # Dynamically preserve any additional domain info keys
            for k in infos_list[0][agent].keys():
                if k not in agent_dict:
                    vals = [i[agent][k] for i in infos_list]
                    if isinstance(vals[0], torch.Tensor):
                        agent_dict[k] = torch.stack(vals)
                    else:
                        agent_dict[k] = vals
            b_infos[agent] = agent_dict

        return b_obs, b_rews, b_terms, b_truncs, b_infos