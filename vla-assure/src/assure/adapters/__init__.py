"""Backend sessions share one interface:

    s = make_session(model_name, model_cfg, variant_name, variant, suite, **kw)
    s.module                  # nn.Module that quantization acts on
    s.report                  # dict describing the loaded variant
    s.rollout_episode(task_id, episode_idx, seed, capture=None, cap_max=0) -> store.Episode
    s.replay(captured_input)  # -> {tap: feature}; used by the sensitivity sweep
    s.close()
"""


def make_session(model_name, model_cfg, variant_name, variant, suite, **kw):
    backend = model_cfg["backend"]
    if backend == "lerobot":
        from .lerobot_adapter import LeRobotSession
        return LeRobotSession(model_name, model_cfg, variant_name, variant, suite, **kw)
    if backend == "openvla":
        from .openvla_adapter import OpenVLASession
        return OpenVLASession(model_name, model_cfg, variant_name, variant, suite, **kw)
    raise ValueError(f"unknown backend {backend}")
