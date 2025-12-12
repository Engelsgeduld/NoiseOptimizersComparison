from experiment.model_trainer import StandardTrainer
from experiment.scheduled_trainer import ScheduledSamplingTrainer

TRAINERS_MAP = {
    "standard": StandardTrainer,
    "sheduled": ScheduledSamplingTrainer,
}
