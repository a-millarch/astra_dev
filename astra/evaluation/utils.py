from astra.utils import logger, cfg
from astra.models.hybrid.training import get_backbone, Learner, patch_learner_get_preds

def prepare_learner(data, model_name=None):
    if model_name is None:
        logger.info(f"Using default model name from cfg: {cfg['model_name']}")
        model_name = cfg["model_name"]

    logger.info(f"Loading model: {model_name}")
    backbone = get_backbone(data, cfg)
    learn = Learner(data["mixed_dls"], backbone, metrics=None)
    learn.load(model_name)
    learn.to('cuda')
    learn = patch_learner_get_preds(learn)
    logger.info("Model loaded")
    return learn
