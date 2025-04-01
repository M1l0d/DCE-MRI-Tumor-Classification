import subprocess
import os

# ---- Config ----
seeds = [42, 1337, 7, 123, 2024]
input_mode = "delta"  # "t2", "delta", or "delta2"
batch_size = 32
epochs = 100
lr = 5e-4

# ---- Paths ----
script_path = "Models/DeepyNetV2_1/run_v2_1.py"

for seed in seeds:
    cmd = [
        "python", script_path,
        "--input_mode", input_mode,
        "--batch_size", str(batch_size),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--seed", str(seed)
    ]

    log_dir = f"logs_v2_1/{input_mode}_seed{seed}"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "run.log")

    with open(log_path, "w") as out:
        print(f"🚀 Launching seed {seed} → log: {log_path}")
        subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
