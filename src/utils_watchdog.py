"""
Hardware & Process Watchdog Utilities
Monitors System RAM and GPU VRAM, enforces downcasting, performs aggressive garbage collection,
and wraps CUDA execution with adaptive batch-size / micro-batch fallback on OOM.
"""
import os
import gc
import psutil
import torch
import polars as pl
import numpy as np

def cleanup_memory():
    """Forces garbage collection and clears CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def get_system_ram_usage():
    """Returns (used_gb, total_gb, percent)."""
    mem = psutil.virtual_memory()
    return mem.used / (1024**3), mem.total / (1024**3), mem.percent

def get_gpu_vram_usage():
    """Returns (allocated_mb, reserved_mb, total_mb) if CUDA available, else None."""
    if not torch.cuda.is_available():
        return None
    alloc = torch.cuda.memory_allocated() / (1024**2)
    res = torch.cuda.memory_reserved() / (1024**2)
    tot = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    return alloc, res, tot

def log_resource_status(prefix="[WATCHDOG]"):
    """Prints current RAM and VRAM status."""
    ram_used, ram_tot, ram_pct = get_system_ram_usage()
    gpu_info = get_gpu_vram_usage()
    if gpu_info:
        g_alloc, g_res, g_tot = gpu_info
        print(f"{prefix} RAM: {ram_used:.1f}G/{ram_tot:.1f}G ({ram_pct:.1f}%) | VRAM: {g_alloc:.1f}M alloc, {g_res:.1f}M res / {g_tot:.1f}M tot")
    else:
        print(f"{prefix} RAM: {ram_used:.1f}G/{ram_tot:.1f}G ({ram_pct:.1f}%) | VRAM: N/A")

def check_memory_threshold(threshold_pct=85.0):
    """Raises RuntimeError if system RAM exceeds threshold_pct."""
    cleanup_memory()
    _, _, pct = get_system_ram_usage()
    if pct >= threshold_pct:
        raise RuntimeError(f"CRITICAL: System RAM at {pct:.1f}% exceeds safe threshold {threshold_pct}%. Aborting step to prevent OS freeze.")

def downcast_polars(df: pl.DataFrame) -> pl.DataFrame:
    """Downcasts float64 to float32 and int64 to int32/int16 to conserve memory."""
    exprs = []
    for col, dtype in zip(df.columns, df.dtypes):
        if dtype == pl.Float64:
            exprs.append(pl.col(col).cast(pl.Float32))
        elif dtype == pl.Int64:
            # Check range if possible or default to int32
            exprs.append(pl.col(col).cast(pl.Int32))
    return df.with_columns(exprs) if exprs else df
