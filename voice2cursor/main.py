import click
from pathlib import Path
import logging
import sys

from .config import load_config, set_config
from .app import run_app

@click.command()
@click.option('--config-file', type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Path to a custom YAML configuration file.")
@click.option('--debug', is_flag=True, help="Enable debug logging.")
def main_cli(config_file, debug):
    """
    Runs the Voice2Cursor application, which transcribes voice to text and sends it to the Cursor editor.
    """
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info("Loading configuration.")
    app_config = load_config(config_file)
    set_config(app_config)
    logging.debug("Configuration loaded.")

    app = run_app()
    app.run() 