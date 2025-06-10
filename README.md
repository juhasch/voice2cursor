# Voice2Cursor

A Python application to transcribe voice to text and send it to the Cursor editor's chat input.

## Features

-   Voice activity detection (VAD) to automatically start and stop recording.
-   Transcription using OpenAI's Whisper model.
-   Pastes transcribed text into Cursor's chat.
-   Menu bar application for easy control.
-   Configurable via a `config.yaml` file.
-   Global hotkey to toggle recording.

## Installation

This project uses `uv` as a package manager.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/voice2cursor.git
    cd voice2cursor
    ```

2.  **Install prerequisites:**

    This project requires PortAudio and FFmpeg. On macOS, you can install them with Homebrew:
    ```bash
    brew install portaudio ffmpeg
    ```

3.  **Create a virtual environment and install dependencies:**

    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install -e .
    ```

## Configuration

Configuration is handled via a `config.yaml` file created in the project root. A default configuration will be created if one does not exist.

Here is an example `config.yaml`:

```yaml
# voice2cursor/config.yaml
audio:
  sample_rate: 16000
  channels: 1
  chunk_size: 512

vad:
  threshold: 0.5
  min_speech_duration_ms: 100
  min_silence_duration_ms: 1000

paste_delay: 0.2

hotkey: "alt+r"
```

## Usage

Once installed, you can run the application from your terminal:

```bash
voice2cursor
```

A status icon will appear in your menu bar. You can click it to start/stop monitoring or use the configured hotkey (default: Alt+R).

## Development

### Running Tests

To run the test suite, use `pytest`:

```bash
uv pip install pytest
pytest
``` 