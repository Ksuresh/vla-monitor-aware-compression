"""LeRobot backend: pi0, pi0.5, SmolVLA (and other LeRobot VLAs) on LIBERO.

Reuses LeRobot's own `rollout()` loop, so environment handling, observation processing and
success detection are identical to `lerobot-eval`. Our only additions are forward hooks and a
thin wrapper around `predict_action_chunk` that records each predicted chunk.
Written against the lerobot 0.6.x API (Oct 2026); pin that version.

Pairing: every episode is identified by (task, episode index, seed). The initial state is forced
to `init_states[trial_init_state(seed, task, episode_index)]` (common.py), the env is seeded with `seed`, and torch's RNG (flow-matching
noise) is reseeded from `seed` at episode start, so variant A and variant B face the same trial.

Shadow mode: `shadow={name: variant_spec}` loads extra copies of the policy (e.g. W4). They see
exactly the observation the controlling policy sees, their features/chunks are logged, and their
actions are never executed. This gives an observation-controlled comparison: the same observations
through a different model. It says nothing about what the shadow would have done in control.
"""
from __future__ import annotations

import copy
import logging
import time
import zlib

import numpy as np
import torch

from ..common import trial_init_state
from ..hooks import FeatureTap
from ..perturb import Perturber, condition_key
from ..store import Episode
from ..variants import apply_variant, checkpoint_for

log = logging.getLogger(__name__)


def _pixels(batch, size: int = 24):
    """Downsampled first camera image from a LeRobot batch -> flat float vector."""
    for k, v in batch.items():
        if "image" in k and torch.is_tensor(v) and v.dim() == 4:
            x = torch.nn.functional.interpolate(v[:1].float(), size=(size, size), mode="area")
            return x.flatten().cpu().numpy()
    return None


def _seed_all(seed: int) -> None:
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LeRobotSession:
    def __init__(self, model_name: str, model_cfg: dict, variant_name: str, variant: dict, suite: str,
                 device: str = "cuda", n_alt_samples: int = 0, condition: str = "nominal",
                 condition_spec: dict | None = None, shadow: dict | None = None, base_seed: int = 7, **_):
        from lerobot.envs.configs import LiberoEnv
        from lerobot.policies import make_pre_post_processors

        self.model_name, self.variant_name, self.suite = model_name, variant_name, suite
        self.base_seed = base_seed
        self.n_alt = n_alt_samples
        self.key = condition_key(suite, condition)
        self.perturb = Perturber(condition_spec)
        self.device = device
        self.model_cfg = model_cfg
        self.policy, self.pcfg, self.ckpt, self.report = self._build(variant_name, variant)
        self.module = self.policy
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.pcfg, pretrained_path=self.ckpt,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.tap = FeatureTap(taps=dict(model_cfg["tap"])).attach(self.policy)
        self._orig_predict = self.policy.predict_action_chunk
        self._state = {"step": 0, "ep": None, "capture": None, "cap_max": 0, "seed": 0}
        self.policy.predict_action_chunk = self._predict_and_log
        if self.perturb.acts:
            self._wrap_select_action()
        self.sampler = {  # must be identical across variants (checked by 03_evaluate)
            "num_steps": getattr(self.pcfg, "num_steps", None), "n_action_steps": getattr(self.pcfg, "n_action_steps", None),
            "chunk_size": getattr(self.pcfg, "chunk_size", None), "alt_samples": n_alt_samples,
            "noise_seeding": "torch reseeded per trial; shadows/alt samples on forked RNG"}
        self._envs = {}
        self.query_seconds: list[float] = []

        # Shadow copies (never control the robot). Same checkpoint family; must share preprocessing.
        self.shadows = {}
        for sname, sspec in (shadow or {}).items():
            pol, _, sck, rep = self._build(sname, sspec)
            tap = FeatureTap(taps=dict(model_cfg["tap"])).attach(pol)
            self.shadows[sname] = (pol, tap, pol.predict_action_chunk)
            self.report.setdefault("shadows", {})[sname] = rep
        _ = LiberoEnv  # imported for side effects of registering the env type

    def _build(self, variant_name: str, variant: dict):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnv
        from lerobot.policies import make_policy

        ckpt = checkpoint_for(self.model_cfg, variant, self.suite)
        pcfg = PreTrainedConfig.from_pretrained(ckpt)
        pcfg.pretrained_path = ckpt
        pcfg.device = self.device
        if self.model_cfg.get("n_action_steps") and hasattr(pcfg, "n_action_steps"):
            pcfg.n_action_steps = int(self.model_cfg["n_action_steps"])
        policy = make_policy(cfg=pcfg, env_cfg=LiberoEnv(task=self.suite, task_ids=[0]))
        policy.eval()
        report = apply_variant(policy, self.model_name, variant)
        report.update(model=self.model_name, variant=variant_name, suite=self.suite, checkpoint=ckpt)
        return policy, pcfg, ckpt, report

    # -- called by LeRobot's select_action whenever its action queue is empty
    def _predict_and_log(self, batch, **kw):
        st = self._state
        batch = self.perturb.torch_batch(batch)
        if st["capture"] is not None and len(st["capture"]) < st["cap_max"]:
            st["capture"].append(copy.deepcopy(batch))
        t0 = time.perf_counter()
        chunk = self._orig_predict(batch, **kw)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.query_seconds.append(time.perf_counter() - t0)
        feats = self.tap.pop() or {}
        alt = None
        if self.n_alt > 0:  # extra flow-matching samples with fresh noise on the same observation
            with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
                torch.manual_seed(st["seed"] * 7919 + st["step"])
                alt = torch.stack([self._orig_predict(batch, **kw) for _ in range(self.n_alt)], 1)
            self.tap.pop()
            alt = alt[0].float().cpu().numpy()
        shadow_out = {}
        for sname, (pol, tap, predict) in self.shadows.items():
            # Forked RNG: the shadow's noise draws must not change the controlling policy's draws,
            # otherwise a run with shadows would not be paired with a run without them.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
                torch.manual_seed(st["seed"] * 104729 + st["step"] + zlib.crc32(sname.encode()))
                sch = predict(batch, **kw)
            shadow_out[sname] = ({k: v[0].numpy() for k, v in (tap.pop() or {}).items()}, sch[0].float().cpu().numpy())
        if st["ep"] is not None:
            feats = {k: v[0].numpy() for k, v in feats.items()}
            ext = _pixels(batch)
            if ext is not None:
                feats["ext_pixels"] = ext  # policy-independent evidence: raw camera, no model weights
            st["ep"].add_query(st["step"], feats, chunk[0].float().cpu().numpy(), alt)
            for sname, (sf, sc) in shadow_out.items():
                st["ep"].add_shadow(sname, sf, sc)
        return chunk

    def _on_step(self, _policy):
        self._state["step"] += 1

    def _wrap_select_action(self):
        """Actuator-noise condition: perturb each EXECUTED action (after the policy picks it from its
        queue); the logged chunk stays the policy's own output, so monitors see what the policy produced."""
        orig = self.policy.select_action

        def select(batch, **kw):
            return self.perturb.action(orig(batch, **kw))
        self.policy.select_action = select

    def _env(self, task_id):
        from lerobot.envs import make_env, make_env_pre_post_processors
        from lerobot.envs.configs import LiberoEnv

        if task_id not in self._envs:
            env_cfg = LiberoEnv(task=self.suite, task_ids=[task_id])
            envs = make_env(env_cfg, n_envs=1)
            vec = envs[self.suite].get(task_id) or next(iter(envs[self.suite].values()))
            self._envs[task_id] = (vec, *make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=self.pcfg))
        return self._envs[task_id]

    @staticmethod
    def _force_init_state(vec, idx: int) -> bool:
        """LeRobot's LiberoEnv advances init_state_id on every reset; pin it so episode i always
        starts from init state i, whatever ran before (skips, autoresets, crashes)."""
        try:
            vec.set_attr("init_state_id", idx)
            return True
        except Exception:  # noqa: BLE001
            try:
                vec.envs[0].unwrapped.init_state_id = idx
                return True
            except Exception:  # noqa: BLE001
                log.warning("could not pin init_state_id; pairing across variants is NOT guaranteed")
                return False

    def rollout_episode(self, task_id: int, episode_idx: int, seed: int, capture: list | None = None,
                        cap_max: int = 0) -> Episode:
        from lerobot.scripts.lerobot_eval import rollout

        vec, env_pre, env_post = self._env(task_id)
        ep = Episode(self.model_name, self.variant_name, self.key, task_id, episode_idx, seed)
        init_id = trial_init_state(self.base_seed, task_id, episode_idx)
        if self._force_init_state(vec, init_id):
            ep.init_state_id = init_id
        self.perturb.reset(seed)
        _seed_all(seed)
        self._state.update(step=0, ep=ep, capture=capture, cap_max=cap_max, seed=seed)
        with torch.no_grad():
            out = rollout(vec, self.policy, env_pre, env_post, self.pre, self.post,
                          seeds=[seed], predicted_latents_callback=self._on_step)
        done = out["done"][0].numpy()
        ep.length = int(np.argmax(done) + 1) if done.any() else len(done)
        ep.success = bool(out["success"][0].any())
        ep.actions = out["action"][0][: ep.length].numpy().tolist()
        self._state.update(ep=None, capture=None)
        return ep

    @torch.no_grad()
    def replay(self, captured) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Re-run one captured model input; return (tapped features, action chunk).
        Fixed noise seed so action differences reflect the weights, not the sampler."""
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
            torch.manual_seed(0)
            chunk = self._orig_predict(captured)
        feats = {k: v[0].numpy() for k, v in (self.tap.pop() or {}).items()}
        return feats, chunk[0].float().cpu().numpy()

    def close(self):
        self.tap.detach()
        for _, tap, _ in self.shadows.values():
            tap.detach()
        for vec, *_ in self._envs.values():
            vec.close()
