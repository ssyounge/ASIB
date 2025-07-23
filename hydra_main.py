import logging
from omegaconf import DictConfig, OmegaConf
import hydra

from main import main as legacy_main


@hydra.main(config_path="configs", config_name="base", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Hydra entry point forwarding the config to ``main.main``."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    # legacy_main expects CLI parsing; pass config via temporary file
    logging.getLogger(__name__).info("Running with Hydra configuration")
    legacy_main(cfg_dict)


if __name__ == "__main__":
    main()
