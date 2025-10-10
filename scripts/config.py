from pydantic import BaseModel


class DataConfig(BaseModel):
    data_path: str


class ModelConfig(BaseModel):
    vocab_size: int
    context_length: int = 256
    d_model: int
    d_ff: int
    rope_theta: float
    num_layers: int
    num_heads: int


class OptimizerConfig(BaseModel):
    betas: tuple[float, float]
    eps: float
    weight_decay: float


class SchedulerConfig(BaseModel):
    max_learning_rate: float
    min_learning_rate: float
    warmup_iters: int
    cosine_cycle_iters: int


class TrainingConfig(BaseModel):
    device: str
    batch_size: int
    num_epochs: int
    log_interval: int
    save_interval: int
    save_dir: str = "checkpoints"
    resume_from: str | None = None


class WandbConfig(BaseModel):
    enabled: bool = True
    project: str = "cs336-basic"
    entity: str | None = None
    name: str | None = None
    tags: list[str] = []


class Config(BaseModel):
    data: DataConfig
    training: TrainingConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    wandb: WandbConfig = WandbConfig()
