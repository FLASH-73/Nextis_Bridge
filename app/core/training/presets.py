# Training presets for SmolVLA
SMOLVLA_PRESETS = {
    "quick": {
        "steps": 10_000,
        "batch_size": 8,
        "save_freq": 5_000,
        "eval_freq": 5_000,
        "description": "Quick test run (~30 min on GPU)"
    },
    "standard": {
        "steps": 100_000,
        "batch_size": 8,
        "save_freq": 20_000,
        "eval_freq": 20_000,
        "description": "Standard training (~5 hours on GPU)"
    },
    "full": {
        "steps": 200_000,
        "batch_size": 16,
        "save_freq": 25_000,
        "eval_freq": 25_000,
        "description": "Full training for best quality (~10 hours on GPU)"
    }
}

# Training presets for Diffusion Policy
DIFFUSION_PRESETS = {
    "quick": {
        "steps": 10_000,
        "batch_size": 32,
        "save_freq": 5_000,
        "eval_freq": 5_000,
        "description": "Quick test (~15 min on GPU)"
    },
    "standard": {
        "steps": 50_000,
        "batch_size": 64,
        "save_freq": 10_000,
        "eval_freq": 10_000,
        "description": "Standard training (~1 hour)"
    },
    "full": {
        "steps": 100_000,
        "batch_size": 64,
        "save_freq": 20_000,
        "eval_freq": 20_000,
        "description": "Full training (~2 hours)"
    }
}

# Training presets for Pi0.5 Policy (with LoRA: ~22GB VRAM, without: ~40GB+)
# Optimized for small datasets (20-50 episodes) to prevent overfitting
PI05_PRESETS = {
    "quick": {
        "steps": 2_000,
        "batch_size": 8,
        "save_freq": 500,
        "eval_freq": 500,
        "description": "Quick test (~10 min, good for debugging)"
    },
    "standard": {
        "steps": 5_000,
        "batch_size": 8,
        "save_freq": 1_000,
        "eval_freq": 1_000,
        "description": "Standard finetuning (best for 20-50 episodes)"
    },
    "full": {
        "steps": 15_000,
        "batch_size": 8,
        "save_freq": 2_500,
        "eval_freq": 2_500,
        "description": "Extended training (50+ episodes, risk of overfitting with less)"
    }
}

# Default training config for SmolVLA
SMOLVLA_DEFAULTS = {
    "pretrained_model": "lerobot/smolvla_base",
    "freeze_vision_encoder": True,
    "train_expert_only": True,
    "train_state_proj": True,
    "learning_rate": 1e-4,
    "warmup_steps": 1000,
    "num_workers": 4,
}

# Default training config for Diffusion Policy
DIFFUSION_DEFAULTS = {
    "n_obs_steps": 2,
    "horizon": 16,
    "n_action_steps": 8,
    "noise_scheduler_type": "DDPM",
    "num_train_timesteps": 100,
    "vision_backbone": "resnet18",
    "learning_rate": 1e-4,
    "warmup_steps": 500,
    "num_workers": 4,
    "resize_shape": [480, 640],  # Default resize for multi-camera support (height, width)
}

# Default training config for Pi0.5 Policy
PI05_DEFAULTS = {
    "pretrained_path": "lerobot/pi05_base",  # or "lerobot/pi05_libero"
    "gradient_checkpointing": True,  # CRITICAL: reduces VRAM significantly
    "dtype": "bfloat16",  # CRITICAL: half precision
    "compile_model": False,  # Disabled - compilation warmup uses extra memory
    "chunk_size": 50,
    "n_action_steps": 50,
    "learning_rate": 2.5e-5,
    "warmup_steps": 1_000,
    "num_workers": 4,
    "use_quantile_normalization": True,  # If False, use MEAN_STD fallback
    # LoRA settings (for memory-efficient fine-tuning)
    "lora_rank": 8,  # 0 = disabled, 8 = recommended for small datasets
    "lora_alpha": 16,  # Scaling factor (typically 2x rank)
    "lora_dropout": 0.1,
}

# Training presets for ACT (Action Chunking with Transformers)
# Lightweight model (~8GB VRAM), designed for bimanual fine manipulation
ACT_PRESETS = {
    "quick": {
        "steps": 50_000,
        "batch_size": 8,
        "save_freq": 10_000,
        "eval_freq": 10_000,
        "description": "Quick test (~20 min on GPU)"
    },
    "standard": {
        "steps": 100_000,
        "batch_size": 8,
        "save_freq": 25_000,
        "eval_freq": 25_000,
        "description": "Standard training (best for 20-50 episodes)"
    },
    "full": {
        "steps": 200_000,
        "batch_size": 8,
        "save_freq": 50_000,
        "eval_freq": 50_000,
        "description": "Extended training for maximum quality"
    }
}

# Default training config for ACT
ACT_DEFAULTS = {
    "chunk_size": 100,
    "n_action_steps": 100,
    "n_obs_steps": 1,
    "use_vae": True,
    "latent_dim": 32,
    "kl_weight": 10.0,
    "vision_backbone": "resnet18",
    "dim_model": 512,
    "learning_rate": 1e-5,
    "warmup_steps": 500,
    "num_workers": 4,
}
