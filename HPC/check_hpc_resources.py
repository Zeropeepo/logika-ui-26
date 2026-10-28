#!/usr/bin/env python3
"""
HPC Pre-Flight Compute & Resource Verification Script
Checks:
- NVIDIA GPU detection (A100 SXM4 80GB target)
- Real-time free VRAM on shared GPU
- CPU core availability & System RAM
- Disk space
- PyTorch CUDA compatibility with NVIDIA driver 580.x
Outputs recommended execution configuration parameters.
"""
import os
import sys
import shutil
import subprocess

def check_resources():
    print("=" * 80)
    print("AI CENTER UB HPC - PRE-FLIGHT COMPUTE & RESOURCE AUDIT")
    print("=" * 80)

    # 1. Python & Environment
    print(f"[*] Python Executable : {sys.executable}")
    print(f"[*] Python Version    : {sys.version.split()[0]}")
    is_venv = sys.prefix != sys.base_prefix
    print(f"[*] Isolated Venv     : {'YES (Isolated user-space)' if is_venv else 'NO (System Python - Caution!)'}")

    # 2. CPU & System RAM
    try:
        import psutil
        cpu_cnt = psutil.cpu_count(logical=True)
        cpu_load = psutil.cpu_percent(interval=0.5)
        ram = psutil.virtual_memory()
        print(f"[*] CPU Cores         : {cpu_cnt} threads (Current load: {cpu_load}%)")
        print(f"[*] System RAM        : Total {ram.total / 1024**3:.1f} GB | Available: {ram.available / 1024**3:.1f} GB ({ram.percent}% used)")
    except ImportError:
        print("[!] psutil not installed. Basic CPU info:")
        cpu_cnt = os.cpu_count() or 1
        print(f"[*] CPU Cores         : {cpu_cnt}")

    # 3. Disk Space
    cwd_disk = shutil.disk_usage(os.getcwd())
    print(f"[*] Storage (CWD)     : Free {cwd_disk.free / 1024**3:.1f} GB / Total {cwd_disk.total / 1024**3:.1f} GB")

    # 4. GPU & CUDA
    print("\n" + "-" * 40 + " GPU & CUDA STATUS " + "-" * 40)
    gpu_ok = False
    free_vram_gb = 0.0
    total_vram_gb = 0.0

    try:
        smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT
        ).decode("utf-8").strip()
        
        lines = smi_out.splitlines()
        for i, line in enumerate(lines):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                name, driver, tot_m, free_m, used_m, util = parts[:6]
                tot_gb = float(tot_m) / 1024.0
                fr_gb = float(free_m) / 1024.0
                us_gb = float(used_m) / 1024.0
                print(f"[*] GPU [{i}] Name       : {name}")
                print(f"[*] Driver Version    : {driver}")
                print(f"[*] VRAM Total        : {tot_gb:.1f} GB")
                print(f"[*] VRAM Free (Shared): {fr_gb:.1f} GB | Used: {us_gb:.1f} GB ({util}% load)")
                if i == 0:
                    gpu_ok = True
                    free_vram_gb = fr_gb
                    total_vram_gb = tot_gb
    except Exception as e:
        print(f"[!] nvidia-smi query failed: {e}")

    # PyTorch CUDA check
    try:
        import torch
        print(f"\n[*] PyTorch Version   : {torch.__version__}")
        print(f"[*] CUDA Available    : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[*] PyTorch CUDA Vers : {torch.version.cuda}")
            print(f"[*] Device Count      : {torch.cuda.device_count()}")
            dev_name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            print(f"[*] Device 0 Name     : {dev_name} (Compute Capability: {cap[0]}.{cap[1]})")
    except ImportError:
        print("[!] PyTorch not yet installed in current environment.")

    # 5. Recommendation Engine
    print("\n" + "=" * 80)
    print("RECOMMENDED PIPELINE CONFIGURATION")
    print("=" * 80)
    if free_vram_gb >= 40.0:
        tier = "A100_HIGH"
        kumo_size = "medium"
        context_size = 4000
        num_estimators = 8
        n_bags = 5
        chunk_size = 10000
        cb_task_type = "GPU"
    elif free_vram_gb >= 15.0:
        tier = "A100_MEDIUM"
        kumo_size = "medium"
        context_size = 2500
        num_estimators = 4
        n_bags = 3
        chunk_size = 5000
        cb_task_type = "GPU"
    elif free_vram_gb >= 6.0:
        tier = "SHARED_LOW"
        kumo_size = "small"
        context_size = 1500
        num_estimators = 2
        n_bags = 2
        chunk_size = 3000
        cb_task_type = "GPU"
    else:
        tier = "FALLBACK_CPU_SAFE"
        kumo_size = "small"
        context_size = 1000
        num_estimators = 2
        n_bags = 1
        chunk_size = 1000
        cb_task_type = "CPU"

    print(f"--> Resource Tier Detected: {tier}")
    print(f"    - KumoTabular Model Size  : '{kumo_size}'")
    print(f"    - Context Points / Bag    : {context_size}")
    print(f"    - Number of Estimators    : {num_estimators}")
    print(f"    - In-Context Bags / Fold  : {n_bags}")
    print(f"    - Query Chunk Size        : {chunk_size}")
    print(f"    - CatBoost Task Type      : '{cb_task_type}'")
    print("=" * 80)

    return {
        "tier": tier,
        "kumo_size": kumo_size,
        "context_size": context_size,
        "num_estimators": num_estimators,
        "n_bags": n_bags,
        "chunk_size": chunk_size,
        "cb_task_type": cb_task_type,
        "free_vram_gb": free_vram_gb
    }

if __name__ == "__main__":
    check_resources()
