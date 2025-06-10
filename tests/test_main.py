import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

# Mock modules that have side effects or are not available in the test environment
modules_to_mock = {
    'pyaudio': MagicMock(),
    'whisper': MagicMock(),
    'torch': MagicMock(),
    'openai': MagicMock(),
    'pynput': MagicMock(),
    'rumps': MagicMock(),
    'pyperclip': MagicMock(),
    'pyautogui': MagicMock()
}

@pytest.fixture(autouse=True)
def mock_modules():
    with patch.dict('sys.modules', modules_to_mock):
        yield

@pytest.fixture
def runner():
    return CliRunner()

def test_app_initialization(runner):
    """
    Tests if the MenuBarApp can be initialized via the CLI without errors.
    """
    # We need to import main after the modules are mocked
    from voice2cursor.main import main_cli
    from voice2cursor.config import set_config, AppConfig
    
    # We explicitly set a config to avoid file loading in tests
    set_config(AppConfig())
    
    # We patch MenuBarApp's run method to prevent it from actually running
    with patch('voice2cursor.main.MenuBarApp.run'):
        result = runner.invoke(main_cli)

    assert result.exit_code == 0
    modules_to_mock['rumps'].App.assert_called_once()

def test_hotkey_parsing():
    """
    Test the hotkey parsing logic from the pydantic model.
    """
    from voice2cursor.config import AppConfig
    from pynput import keyboard

    config = AppConfig(hotkey='cmd+shift+s')
    hotkey = config.parsed_hotkey
    assert hotkey == [keyboard.Key.cmd, keyboard.Key.shift, keyboard.KeyCode.from_char('s')] 