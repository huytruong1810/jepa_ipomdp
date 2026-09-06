# ABSOLUTE PATH: src/ipomdp/envs/tiger.py
# ==============================================================================
# CANONICAL MULTI-AGENT TIGER I-POMDP DOMAIN & EXACT BAYESIAN ORACLE
# ==============================================================================
#
# DESIGN DECISIONS & THEORETICAL FOUNDATIONS:
# 1. Strict Canonical POMDP Benchmark Formulation:
#    - Conforms strictly to Kaelbling et al. (1998) POMDP and Doshi & Gmytrasiewicz (2006)
#      I-POMDP standards without non-canonical artifacts.
#    - States: S = {Tiger-Left: 0, Tiger-Right: 1}. Initial state s_0 ~ Uniform({TL, TR}).
#    - Actions: A_i = A_j = {LISTEN: 0, OPEN_LEFT: 1, OPEN_RIGHT: 2}.
#    - Observations: (Growl, Creak) where Growl in {GL: 0.0, GR: 1.0} with accuracy alpha=0.85,
#      and Creak in {SILENCE: -1.0, CREAK_L: 0.0, CREAK_R: 1.0} with accuracy beta=1.0.
#    - Payoffs: Listen (-1.0), Gold (+10.0), Tiger (-100.0).
#
# 2. Canonical 6-Class Discrete Observation Partition:
#    - Maps joint continuous observation vectors [growl, creak] into 6 canonical classes:
#         0: (GROWL_LEFT,  SILENCE)
#         1: (GROWL_RIGHT, SILENCE)
#         2: (GROWL_LEFT,  CREAK_LEFT)
#         3: (GROWL_LEFT,  CREAK_RIGHT)
#         4: (GROWL_RIGHT, CREAK_LEFT)
#         5: (GROWL_RIGHT, CREAK_RIGHT)
#
# 3. Exact Analytical Bayesian Belief Filter & Observation Distribution Oracle:
#    - Tracks exact scalar probability b_t*(TL) = P(s_t = TL | o_{1:t}, a_{1:t-1}).
#    - Evaluates exact double-precision conditional distribution P*(o_{t+1} | b_t*, a_t, a_{-i})
#      summing strictly to 1.0.
# ==============================================================================

from typing import Dict, Tuple, List, Optional
import torch

from ..interfaces import IPOMDPEnv
from ..types import AgentID, State, Observation, Action, StepResult
from ..telemetry.registry import register_env

# Discrete Action Constants
LISTEN = 0
OPEN_LEFT = 1
OPEN_RIGHT = 2

# Discrete State Constants
TIGER_LEFT = 0
TIGER_RIGHT = 1

# Canonical Discrete Observation Signals
GROWL_LEFT = 0.0
GROWL_RIGHT = 1.0

SILENCE = -1.0
CREAK_LEFT = 0.0
CREAK_RIGHT = 1.0

# Canonical 6 Discrete Observation Categories: (Growl, Creak)
OBS_CATEGORIES: List[Tuple[float, float]] = [
    (0.0, -1.0),  # 0: GROWL_LEFT, SILENCE
    (1.0, -1.0),  # 1: GROWL_RIGHT, SILENCE
    (0.0, 0.0),   # 2: GROWL_LEFT, CREAK_LEFT
    (0.0, 1.0),   # 3: GROWL_LEFT, CREAK_RIGHT
    (1.0, 0.0),   # 4: GROWL_RIGHT, CREAK_LEFT
    (1.0, 1.0),   # 5: GROWL_RIGHT, CREAK_RIGHT
]
NUM_OBS_CLASSES = len(OBS_CATEGORIES)


def encode_observation_to_index(obs_tensor: torch.Tensor) -> int:
    """
    Maps continuous observation vector [growl, creak] to discrete integer class index [0..5].

    Args:
        obs_tensor: Observation tensor of shape (2,) or (1, 2).

    Returns:
        Integer category index in range [0..5].
    """
    vals = obs_tensor.view(-1).tolist()
    if len(vals) != 2:
        raise ValueError(
            f"encode_observation_to_index expected 2 elements [growl, creak], got shape {obs_tensor.shape}"
        )
    growl, creak = vals[0], vals[1]

    # Exact match lookup
    for idx, (g, c) in enumerate(OBS_CATEGORIES):
        if abs(growl - g) < 1e-3 and abs(creak - c) < 1e-3:
            return idx

    # Nearest neighbor fallback for noisy float inputs
    growl_bit = 0.0 if growl < 0.5 else 1.0
    if creak < -0.5:
        return 0 if growl_bit == 0.0 else 1
    creak_bit = 0.0 if creak < 0.5 else 1.0

    if growl_bit == 0.0 and creak_bit == 0.0:
        return 2
    if growl_bit == 0.0 and creak_bit == 1.0:
        return 3
    if growl_bit == 1.0 and creak_bit == 0.0:
        return 4
    return 5


def decode_index_to_observation(class_idx: int) -> torch.Tensor:
    """
    Decodes integer class index back to canonical [growl, creak] observation tensor.

    Args:
        class_idx: Integer class index in range [0..5].

    Returns:
        Observation tensor of shape (2,).
    """
    g, c = OBS_CATEGORIES[class_idx]
    return torch.tensor([g, c], dtype=torch.float32)


@register_env("tiger")
class MultiAgentTigerEnv(IPOMDPEnv):
    """
    Canonical Multi-Agent Tiger Environment (Kaelbling et al. / Doshi & Gmytrasiewicz standard).
    Persistent Variant: Opening a door stochastically relocates the tiger (s' ~ Uniform({TL, TR}))
    and emits a canonical noisy growl o ~ O(s') without truncating the episode.
    """

    def __init__(self, growl_accuracy: float = 0.85, creak_accuracy: float = 1.0, max_steps: int = 20):
        """
        Initializes Tiger environment dynamics and observation noise parameters.

        Args:
            growl_accuracy: Growl emission fidelity alpha (default: 0.85).
            creak_accuracy: Door creak emission fidelity (default: 1.0).
            max_steps: Episode truncation horizon.
        """
        super().__init__()
        self.agents = ["agent_0", "agent_1"]
        self.growl_acc = float(growl_accuracy)
        self.creak_acc = float(creak_accuracy)
        self.max_steps = int(max_steps)
        self.current_step = 0
        self._current_state = State(torch.zeros(1, dtype=torch.float32))

    def reset(self) -> Tuple[Dict[AgentID, Observation], Dict[AgentID, dict]]:
        """
        Initializes episode step count, samples initial state s_0 ~ Uniform({TL, TR}),
        and emits canonical initial start observations o_0 ~ O(s_0, LISTEN, LISTEN).

        Returns:
            Tuple of (initial_observation_dict, initial_info_dict).
        """
        self.current_step = 0
        self._current_state = State(torch.randint(0, 2, (1,), dtype=torch.float32))

        # Initial canonical observation emitted directly from initial state s_0
        dummy_actions = {agent: Action(torch.tensor([LISTEN], dtype=torch.float32)) for agent in self.agents}
        obs = self._get_observation(self._current_state, dummy_actions)
        infos = {agent: {"true_state": self._current_state} for agent in self.agents}
        return obs, infos

    def step(self, actions: Dict[AgentID, Action]) -> StepResult:
        """
        Advances environment simulation by one discrete timestep given joint agent actions.

        Args:
            actions: Dictionary mapping AgentID to Action containers.

        Returns:
            StepResult containing observations, rewards, terminations, truncations, and infos.
        """
        self.current_step += 1
        current_state = self._current_state

        # Transition Dynamics: T(s, a, s')
        next_state = self._transition_dynamics(current_state, actions)

        # Reward Function: R(s, a)
        rewards = self._get_reward(current_state, actions)

        # Observation Function: O(s', a, o) conditioned strictly on s_{t+1}
        observations = self._get_observation(next_state, actions)

        self._current_state = next_state
        is_truncated = self.current_step >= self.max_steps

        terminations = {agent: torch.tensor([False], dtype=torch.bool) for agent in self.agents}
        truncations = {agent: torch.tensor([is_truncated], dtype=torch.bool) for agent in self.agents}
        infos = {agent: {"true_state": next_state} for agent in self.agents}

        rewards_t = {agent: torch.tensor([r], dtype=torch.float32) for agent, r in rewards.items()}

        return StepResult(observations, rewards_t, terminations, truncations, infos)

    def _transition_dynamics(self, state: State, actions: Dict[AgentID, Action]) -> State:
        """
        State Transition Dynamics T(s, a, s'):
        - If ANY agent opens a door, s' ~ Uniform({TL, TR}).
        - If ALL agents listen, s' = s.
        """
        opens = {
            agent: int(actions[agent].data.view(-1)[0].item()) in [OPEN_LEFT, OPEN_RIGHT]
            for agent in self.agents
        }
        door_opened = any(opens.values())

        if door_opened:
            return State(torch.randint(0, 2, (1,), dtype=torch.float32))
        return State(state.data.clone())

    def _get_observation(self, state: State, actions: Dict[AgentID, Action]) -> Dict[AgentID, Observation]:
        """
        Observation Function O(s', a, o):
        Emits binary growl conditioned on tiger position s' with accuracy growl_acc.
        Emits door creak conditioned on opponent action with precision creak_acc.
        """
        observations = {}
        tiger_pos = int(state.data.view(-1)[0].item())

        for agent_id in self.agents:
            other_id = "agent_1" if agent_id == "agent_0" else "agent_0"
            other_action = int(actions[other_id].data.view(-1)[0].item())

            # Growl signal generation
            base_growl = GROWL_LEFT if tiger_pos == TIGER_LEFT else GROWL_RIGHT
            if torch.rand(1).item() > self.growl_acc:
                val_growl = GROWL_RIGHT if base_growl == GROWL_LEFT else GROWL_LEFT
            else:
                val_growl = base_growl

            # Creak signal generation
            if torch.rand(1).item() < self.creak_acc:
                val_creak = CREAK_LEFT if other_action == OPEN_LEFT else (
                    CREAK_RIGHT if other_action == OPEN_RIGHT else SILENCE
                )
            else:
                rand_idx = torch.randint(0, 3, (1,)).item()
                val_creak = [CREAK_LEFT, CREAK_RIGHT, SILENCE][rand_idx]

            observations[agent_id] = Observation(data=torch.tensor([val_growl, val_creak], dtype=torch.float32))

        return observations

    def _get_reward(
        self,
        state: State,
        actions: Dict[AgentID, Action],
        next_state: Optional[State] = None
    ) -> Dict[AgentID, float]:
        """
        Standard POMDP Reward Function R(s, a):
        - Listen: -1.0
        - Gold (Safe Door): +10.0
        - Tiger (Fatal Door): -100.0
        """
        rewards = {}
        tiger_pos = int(state.data.view(-1)[0].item())

        for agent_id in self.agents:
            act = int(actions[agent_id].data.view(-1)[0].item())
            if act == LISTEN:
                rewards[agent_id] = -1.0
            elif act == OPEN_LEFT:
                rewards[agent_id] = -100.0 if tiger_pos == TIGER_LEFT else 10.0
            elif act == OPEN_RIGHT:
                rewards[agent_id] = -100.0 if tiger_pos == TIGER_RIGHT else 10.0
            else:
                rewards[agent_id] = -1.0
        return rewards

    def render(self, actions: Dict[AgentID, Action], step_results: StepResult) -> str:
        """Renders environment state, joint actions, observations, and rewards to text format."""
        result = ""
        observations = step_results.observations
        rewards = step_results.rewards
        infos = step_results.infos

        true_state = int(infos["agent_0"]["true_state"].data.view(-1)[0].item())
        true_state_str = "TIGER_LEFT" if true_state == TIGER_LEFT else "TIGER_RIGHT"

        for agent_id in self.agents:
            action = int(actions[agent_id].data.view(-1)[0].item())
            obs = observations[agent_id].data.view(-1).tolist()
            reward = rewards[agent_id].view(-1)[0].item()

            action_str = "LISTEN" if action == LISTEN else ("OPEN_LEFT" if action == OPEN_LEFT else "OPEN_RIGHT")
            growl_str = "GROWL_LEFT" if obs[0] == GROWL_LEFT else "GROWL_RIGHT"

            if abs(obs[1] - CREAK_LEFT) < 1e-3:
                creak_str = "CREAK_LEFT"
            elif abs(obs[1] - CREAK_RIGHT) < 1e-3:
                creak_str = "CREAK_RIGHT"
            else:
                creak_str = "SILENCE"

            result += f"[{agent_id}] Act: {action_str:<10} | Obs: ({growl_str}, {creak_str}) | Reward: {reward:+.2f}\n"

        result += f"[Truth] {true_state_str}"
        return result


class TigerBayesianOracle:
    """
    Exact Analytical Bayesian Belief Filter and Ground-Truth Observation Oracle.
    Calculates exact P*(s_t) and analytical conditional observation distributions P*(o_{t+1} | b_t*, a_t).
    """

    def __init__(self, growl_accuracy: float = 0.85, creak_accuracy: float = 1.0):
        """
        Initializes Bayesian Oracle.

        Args:
            growl_accuracy: Growl emission fidelity alpha (default: 0.85).
            creak_accuracy: Door creak precision (default: 1.0).
        """
        self.growl_acc = float(growl_accuracy)
        self.creak_acc = float(creak_accuracy)
        self.belief_tiger_left = 0.5  # b_0*(s=TL) = 0.5 (Uniform Start Prior)

    def reset(self) -> float:
        """Resets scalar Bayesian belief to uniform prior b_0 = 0.5."""
        self.belief_tiger_left = 0.5
        return self.belief_tiger_left

    def update(self, action: int, obs_tensor: torch.Tensor) -> float:
        """
        Updates scalar Bayesian belief state b_{t+1}* given action a_t and observation o_{t+1}.
        For door opening actions, belief resets to uniform 0.5 before processing post-reset growl.

        Args:
            action: Discrete ego action taken at step t.
            obs_tensor: Observation tensor received at step t+1.

        Returns:
            Updated scalar belief probability P(s_{t+1} = TL).
        """
        if action in [OPEN_LEFT, OPEN_RIGHT]:
            b_prior = 0.5
        else:
            b_prior = self.belief_tiger_left

        obs_idx = encode_observation_to_index(obs_tensor)

        # Likelihood P(o_growl | s)
        if obs_idx in [0, 2, 3]:
            # GROWL_LEFT received
            p_obs_given_tl = self.growl_acc
            p_obs_given_tr = 1.0 - self.growl_acc
        else:
            # GROWL_RIGHT received
            p_obs_given_tl = 1.0 - self.growl_acc
            p_obs_given_tr = self.growl_acc

        num = p_obs_given_tl * b_prior
        den = p_obs_given_tl * b_prior + p_obs_given_tr * (1.0 - b_prior)

        self.belief_tiger_left = num / max(den, 1e-12)
        return self.belief_tiger_left

    def get_exact_observation_distribution(self, action: int, opponent_action: int = LISTEN) -> torch.Tensor:
        """
        Returns exact probability distribution P*(o_{t+1} | b_t*, a_t, a_{-i}) over all 6 observation classes.

        Args:
            action: Discrete ego action index.
            opponent_action: Discrete opponent action index.

        Returns:
            Double-precision probability tensor of shape (6,) summing to 1.0.
        """
        p_dist = torch.zeros(NUM_OBS_CLASSES, dtype=torch.float64)

        if action in [OPEN_LEFT, OPEN_RIGHT]:
            b_tl = 0.5
        else:
            b_tl = self.belief_tiger_left

        b_tr = 1.0 - b_tl

        # Marginal growl probabilities
        p_growl_left = b_tl * self.growl_acc + b_tr * (1.0 - self.growl_acc)
        p_growl_right = b_tl * (1.0 - self.growl_acc) + b_tr * self.growl_acc

        if opponent_action == LISTEN:
            p_silence = self.creak_acc + (1.0 - self.creak_acc) / 3.0
            p_creak_l = (1.0 - self.creak_acc) / 3.0
            p_creak_r = (1.0 - self.creak_acc) / 3.0
        elif opponent_action == OPEN_LEFT:
            p_creak_l = self.creak_acc + (1.0 - self.creak_acc) / 3.0
            p_silence = (1.0 - self.creak_acc) / 3.0
            p_creak_r = (1.0 - self.creak_acc) / 3.0
        else:  # OPEN_RIGHT
            p_creak_r = self.creak_acc + (1.0 - self.creak_acc) / 3.0
            p_silence = (1.0 - self.creak_acc) / 3.0
            p_creak_l = (1.0 - self.creak_acc) / 3.0

        # Joint observation probability P(growl, creak) = P(growl) * P(creak)
        p_dist[0] = p_growl_left * p_silence   # GROWL_LEFT, SILENCE
        p_dist[1] = p_growl_right * p_silence  # GROWL_RIGHT, SILENCE
        p_dist[2] = p_growl_left * p_creak_l   # GROWL_LEFT, CREAK_LEFT
        p_dist[3] = p_growl_left * p_creak_r   # GROWL_LEFT, CREAK_RIGHT
        p_dist[4] = p_growl_right * p_creak_l  # GROWL_RIGHT, CREAK_LEFT
        p_dist[5] = p_growl_right * p_creak_r  # GROWL_RIGHT, CREAK_RIGHT

        return p_dist

    def predict_most_likely_observation(self, action: int, opponent_action: int = LISTEN) -> int:
        """Returns top-1 argmax observation category index under exact Bayesian dynamics."""
        p_dist = self.get_exact_observation_distribution(action, opponent_action)
        return torch.argmax(p_dist).item()
