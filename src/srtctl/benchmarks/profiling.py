# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profiling benchmark runner for torch/nsys profiling."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, BenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


@register_benchmark("profiling")
class ProfilingRunner(BenchmarkRunner):
    """Profiling benchmark runner.

    Sends /start_profile API calls to workers and generates traffic
    to produce profiling data.

    This benchmark is auto-selected when profiling.type is "torch", "nsys", or "nsys-trace".

    Required config fields (in profiling section):
        - profiling.concurrency: Batch size for profiling
        - profiling.prefill/decode/aggregated: Phase-specific step configs

    For nsys/torch modes (synthetic traffic):
        - profiling.isl: Input sequence length
        - profiling.osl: Output sequence length

    For nsys-trace mode (trace-replay traffic):
        - profiling.trace_file: Path to trace file (JSONL format)
    """

    # Container mount point for trace files (nsys-trace mode)
    TRACE_MOUNT_PATH = Path("/profile-trace-data")

    @property
    def name(self) -> str:
        return "Profiling"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/profiling/profile-trace.sh"

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "profiling")

    def _get_script_path(self, config: SrtConfig) -> str:
        if config.profiling.is_nsys_trace:
            return "/srtctl-benchmarks/profiling/profile-trace.sh"
        return "/srtctl-benchmarks/profiling/profile.sh"

    def _resolve_trace_path(self, trace_file: str) -> Path:
        trace_path = Path(trace_file)
        if trace_path.is_absolute():
            return trace_path
        return Path.cwd() / trace_path

    def get_extra_mounts(self, config: SrtConfig) -> dict[Path, Path]:
        if not config.profiling.is_nsys_trace or config.profiling.trace_file is None:
            return {}
        trace_path = self._resolve_trace_path(config.profiling.trace_file)
        trace_dir = trace_path.parent.resolve()
        return {trace_dir: self.TRACE_MOUNT_PATH}

    def validate_config(self, config: SrtConfig) -> list[str]:
        errors = []
        p = config.profiling

        if not p.enabled:
            errors.append("profiling.type must be 'torch', 'nsys', or 'nsys-trace' for profiling benchmark")
        if p.concurrency is None:
            errors.append("profiling.concurrency is required")

        if p.is_nsys_trace:
            if p.trace_file is None:
                errors.append("profiling.trace_file is required for nsys-trace mode")
            else:
                trace_path = self._resolve_trace_path(p.trace_file)
                if not trace_path.exists():
                    errors.append(f"Trace file not found: {trace_path}")
        else:
            if p.isl is None:
                errors.append("profiling.isl is required")
            if p.osl is None:
                errors.append("profiling.osl is required")

        # Phase config validation is already done in SrtConfig.__post_init__
        return errors

    def build_command(
        self,
        config: SrtConfig,
        runtime: RuntimeContext,
    ) -> list[str]:
        r = config.resources
        script = self._get_script_path(config)

        return [
            "bash",
            script,
            str(r.num_prefill),
            str(r.num_decode),
            str(r.prefill_gpus),
            str(r.decode_gpus),
            str(r.prefill_gpus + r.decode_gpus),
        ]
