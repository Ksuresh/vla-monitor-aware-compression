"""OpenVLA / OpenVLA-OFT backend on LIBERO.

Runs inside the official repo's environment and reuses its helpers (env construction, image
preprocessing, action post-processing) so success rates should match the published eval.
Pass the repo root with --repo_root (openvla/ for base OpenVLA, openvla-oft/ for OFT).

Pairing: episode i always starts from LIBERO init state i and all RNGs are reseeded from the
episode seed. Shadow mode loads extra model copies that see the same observations and are
logged but never executed (see lerobot_adapter for the rationale). Two 7B copies need ~30 GB
in bf16, so run shadows on an 80 GB GPU or one shadow at a time.
"""
from __future__ import annotations

import copy
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..common import trial_init_state
from ..hooks import FeatureTap
from ..perturb import Perturber, condition_key
from ..store import Episode
from ..variants import apply_variant, checkpoint_for

log = logging.getLogger(__name__)

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
WAIT_STEPS = 10  # let objects settle, as in the official eval


def _pixels(img: np.ndarray, size: int = 24) -> np.ndarray:
    """Area-downsample an HxWx3 uint8 image to size x size and flatten to [0,1] floats."""
    h, w = img.shape[:2]
    img = img[: h - h % size, : w - w % size].astype(np.float32) / 255.0
    return img.reshape(size, img.shape[0] // size, size, img.shape[1] // size, -1).mean((1, 3)).ravel()


@dataclass
class _Bundle:
    cfg: object
    model: object
    processor: object
    action_head: object
    proprio: object
    noisy: object
    tap: FeatureTap
    report: dict


class OpenVLASession:
    def __init__(self, model_name: str, model_cfg: dict, variant_name: str, variant: dict, suite: str,
                 repo_root: str, device: str = "cuda", condition: str = "nominal",
                 condition_spec: dict | None = None, shadow: dict | None = None, base_seed: int = 7, **_):
        self.base_seed = base_seed
        self.sampler = {"num_steps": None, "n_action_steps": None, "chunk_size": None, "alt_samples": 0,
                        "noise_seeding": "deterministic decoding; set_seed_everywhere per trial"}
        sys.path.insert(0, str(Path(repo_root).resolve()))
        from libero.libero import benchmark

        from experiments.robot.libero import run_libero_eval as R
        from experiments.robot.robot_utils import get_image_resize_size

        self.R = R
        self.model_name, self.variant_name, self.suite = model_name, variant_name, suite
        self.model_cfg = model_cfg
        self.oft = bool(model_cfg.get("oft"))
        self.key = condition_key(suite, condition)
        self.perturb = Perturber(condition_spec)
        self.main = self._build(variant_name, variant)
        self.module, self.tap, self.report = self.main.model, self.main.tap, self.main.report
        self.ckpt = self.report["checkpoint"]
        self.shadows = {n: self._build(n, s) for n, s in (shadow or {}).items()}
        if self.shadows:
            self.report["shadows"] = {n: b.report for n, b in self.shadows.items()}
        self.resize = get_image_resize_size(self.main.cfg)
        self.suite_obj = benchmark.get_benchmark_dict()[suite]()
        self._envs = {}
        self.query_seconds: list[float] = []

    def _build(self, variant_name: str, variant: dict) -> _Bundle:
        from experiments.robot.robot_utils import get_model
        R = self.R
        ckpt = checkpoint_for(self.model_cfg, variant, self.suite)
        kw = dict(model_family="openvla", pretrained_checkpoint=ckpt,
                  center_crop=self.model_cfg.get("center_crop", True), task_suite_name=self.suite)
        if variant.get("kind") == "bnb":  # real bitsandbytes path -> true memory/latency numbers
            kw.update(load_in_8bit=variant["bits"] == 8, load_in_4bit=variant["bits"] == 4)
        cfg = R.GenerateConfig(**kw)
        if self.oft:
            # initialize_model() sets cfg.unnorm_key itself (adds _no_noops when needed)
            model, head, proprio, noisy, processor = R.initialize_model(cfg)
        else:
            from experiments.robot.openvla_utils import get_processor
            cfg.unnorm_key = self.suite  # mirrors the official eval_libero()
            model = get_model(cfg)
            if self.suite not in model.norm_stats and f"{self.suite}_no_noops" in model.norm_stats:
                cfg.unnorm_key = f"{self.suite}_no_noops"
            processor, head, proprio, noisy = get_processor(cfg), None, None, None
        report = apply_variant(model, self.model_name, variant)
        report.update(model=self.model_name, variant=variant_name, suite=self.suite, checkpoint=ckpt)
        taps = dict(self.model_cfg["tap"])
        tap = FeatureTap(taps=taps, mode={k: "mean" for k in taps}).attach(model)
        return _Bundle(cfg, model, processor, head, proprio, noisy, tap, report)

    def _query(self, b: _Bundle, obs, task_desc) -> np.ndarray:
        from experiments.robot.robot_utils import get_action

        if self.oft:
            observation, _ = self.R.prepare_observation(obs, self.resize)
            acts = get_action(b.cfg, b.model, observation, task_desc, processor=b.processor,
                              action_head=b.action_head, proprio_projector=b.proprio,
                              noisy_action_projector=b.noisy, use_film=b.cfg.use_film)
            return np.stack([np.asarray(a) for a in acts])
        from experiments.robot.libero.libero_utils import get_libero_image, quat2axisangle
        observation = {
            "full_image": get_libero_image(obs, self.resize),
            "state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                     obs["robot0_gripper_qpos"])),
        }
        return np.asarray(get_action(b.cfg, b.model, observation, task_desc, processor=b.processor))[None]

    def _perturbed(self, obs):
        if not self.perturb.active:
            return obs
        obs = dict(obs)
        for k in ("agentview_image", "robot0_eye_in_hand_image"):
            if k in obs:
                obs[k] = self.perturb.numpy_image(obs[k])
        return obs

    @staticmethod
    def _to_env(a):
        from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action
        a = normalize_gripper_action(np.array(a, dtype=np.float64), binarize=True)
        return invert_gripper_action(a)

    def _env(self, task_id):
        from experiments.robot.libero.libero_utils import get_libero_env
        if task_id not in self._envs:
            task = self.suite_obj.get_task(task_id)
            env, desc = get_libero_env(task, "openvla", resolution=256)
            self._envs[task_id] = (env, desc, self.suite_obj.get_task_init_states(task_id))
        return self._envs[task_id]

    def rollout_episode(self, task_id: int, episode_idx: int, seed: int, capture: list | None = None,
                        cap_max: int = 0) -> Episode:
        from experiments.robot.libero.libero_utils import get_libero_dummy_action
        from experiments.robot.robot_utils import set_seed_everywhere

        env, desc, init_states = self._env(task_id)
        ep = Episode(self.model_name, self.variant_name, self.key, task_id, episode_idx, seed)
        ep.init_state_id = trial_init_state(self.base_seed, task_id, episode_idx, len(init_states))
        set_seed_everywhere(seed)
        self.perturb.reset(seed)
        env.seed(seed)
        env.reset()
        obs = env.set_init_state(init_states[ep.init_state_id])
        for _ in range(WAIT_STEPS):
            obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
        queue, t, done = [], 0, False
        while t < MAX_STEPS[self.suite]:
            if not queue:
                pobs = self._perturbed(obs)
                if capture is not None and len(capture) < cap_max:
                    capture.append((copy.deepcopy(pobs), desc))
                t0 = time.perf_counter()
                with torch.no_grad():
                    chunk = self._query(self.main, pobs, desc)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self.query_seconds.append(time.perf_counter() - t0)
                feats = {k: v[0].numpy() for k, v in (self.tap.pop() or {}).items()}
                feats["ext_pixels"] = _pixels(pobs["agentview_image"])  # policy-independent evidence
                ep.add_query(t, feats, chunk)
                for sname, sb in self.shadows.items():
                    with torch.no_grad():
                        sch = self._query(sb, pobs, desc)
                    ep.add_shadow(sname, {k: v[0].numpy() for k, v in (sb.tap.pop() or {}).items()}, sch)
                queue = list(chunk)
            a = self.perturb.action(self._to_env(queue.pop(0)))  # actuator noise condition (no-op otherwise)
            ep.actions.append(np.asarray(a, dtype=np.float32).tolist())
            obs, _, done, _ = env.step(a.tolist())
            t += 1
            if done:
                break
        ep.success, ep.length = bool(done), t
        return ep

    @torch.no_grad()
    def replay(self, captured) -> tuple[dict[str, np.ndarray], np.ndarray]:
        obs, desc = captured
        chunk = self._query(self.main, obs, desc)
        return {k: v[0].numpy() for k, v in (self.tap.pop() or {}).items()}, chunk

    def close(self):
        self.tap.detach()
        for b in self.shadows.values():
            b.tap.detach()
        for env, *_ in self._envs.values():
            env.close()
