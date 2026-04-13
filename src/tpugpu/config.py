from dataclasses import dataclass


@dataclass
class ExpertTrainConfig:
    image_size: int = 32
    num_channels: int = 1
    num_classes: int = 10
    batch_size: int = 128
    num_epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_diffusion_steps: int = 1000
    hidden_channels: int = 64
    checkpoint_dir: str = "./outputs/checkpoints"
    artifact_dir: str = "./outputs/experiments"
    log_every_steps: int = 50
    checkpoint_every_epochs: int = 1
    sample_every_epochs: int = 1
    eval_num_real: int = 512
    eval_num_generated: int = 512
    eval_batch_size: int = 128
    resume: bool = False
    cluster_assignments_path: str | None = None
    cluster_id: int | None = None
    seed: int = 0
    expert_name: str = "expert"
    class_ids: tuple[int, ...] = (0, 1, 2, 3, 4)
