import subprocess

command = [
    "mlx_lm.lora",

    "--model",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",

    "--train",

    "--data",
    "./data/friendly_ict",

    "--adapter-path",
    "./adapters/friendly_ict",

    "--batch-size",
    "1",

    "--iters",
    "300",

    "--learning-rate",
    "1e-4",

    "--max-seq-length",
    "1024",

    "--grad-checkpoint",

    "--val-batches",
    "0",

    "--num-layers",
    "8",

    "--clear-cache-threshold",
    "0.8",
]

subprocess.run(command)