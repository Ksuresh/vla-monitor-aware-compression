# Monitor-Aware VLA Compression

Research repository for testing whether monitor-aware mixed precision preserves failure-monitor reliability better than action-aware mixed precision at matched memory budgets.

## Existing code assessment

The supplied `vla-assure.zip` already contains a substantial Python experiment implementation: LeRobot/OpenVLA adapters, feature hooks, quantization, monitors, conformal calibration, sensitivity-based selection, evaluation, reporting, and 36 unit/integration tests.

The uploaded archive was tested locally with `PYTHONPATH=src pytest -q tests/test_pipeline.py`: **36 tests passed**. This verifies synthetic/unit code paths, **not** actual VLA or LIBERO experiments. No VLA benchmark results exist yet.

## Integration plan

1. Import the supplied `vla-assure/` source tree as the initial codebase, preserving paths.
2. Add a CI workflow to run `PYTHONPATH=src pytest -q tests/test_pipeline.py` on a CPU runner.
3. Check real checkpoint compatibility and LIBERO execution before making empirical claims.
4. Run one SmolVLA + one monitor comparison of action-aware versus monitor-aware allocations at matched parameter budgets.
5. Add independent holdout monitor, executed-trajectory metrics and real INT4 efficiency checks.

**Caution:** Fake INT4 quantization does not establish actual GPU memory or latency savings. Calibration, selection and test data must remain disjoint.
