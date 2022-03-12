from omegaconf import DictConfig, OmegaConf

import hydra
import logging
import os
import sys
import wandb

from main import main
from chexpert_main import main as chexpert_main


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level

    def write(self, buf):
       for line in buf.rstrip().splitlines():
          self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


log = logging.getLogger(__name__)
sys.stderr = StreamToLogger(log, logging.ERROR)  # log stderr to logger


# use hydra for config / savings outputs
@hydra.main(config_path=".", config_name="config")
def setup(flags : DictConfig):
    if os.path.exists("config.yaml"):
        # this lets us requeue runs without worrying if we changed our local config since then
        logging.info("loading pre-existing configuration, we're continuing a previous run")
        new_flags = OmegaConf.load("config.yaml")
        cli_conf = OmegaConf.from_cli()
        # however, you can override parameters from the cli still
        # this is useful e.g. if you did total_epochs=N before and want to increase it
        flags = OmegaConf.merge(new_flags, cli_conf)

    # log config + save it to local directory
    log.info(OmegaConf.to_yaml(flags))
    OmegaConf.save(flags, "config.yaml")

    if flags.wandb:
        wandb.init(project=flags.wbproject, entity=flags.wbentity, group=flags.group, config=flags)
    
    if flags.task == "chexpert":
        return chexpert_main(flags)
    else:
        return main(flags)


if __name__ == "__main__":
    setup()