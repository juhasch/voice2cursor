import logging
import warnings
import threading
import time
import os
import queue
import pyaudio
import wave
import openai
import pyperclip
import pyautogui
import tempfile
import torch
import numpy as np
import whisper
import rumps

from . import config as config_module

# --- Constants ---
INT16_TO_FLOAT_SCALE = 32768.0
FORMAT_MAP = {'paInt16': pyaudio.paInt16}

class MenuBarApp(rumps.App):
    def __init__(self):
        logging.debug("MenuBarApp: Initializing.")
        super().__init__("⚫️ Ready", quit_button=None)
        self.is_monitoring = False
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        self.model_vad = None
        self.model_whisper = None
        self.client = None
        self.loading_queue = queue.Queue()

        self.toggle_button = rumps.MenuItem("Start Recording", callback=self.toggle_monitoring)
        self.menu = [self.toggle_button, None, rumps.MenuItem("Exit", callback=self.on_exit)]

        self.loading_thread = threading.Thread(target=self.load_models, daemon=True)
        self.loading_thread.start()
        self.polling_timer = rumps.Timer(self.check_loading_status, 0.2)
        self.polling_timer.start()

    def check_loading_status(self, sender):
        try:
            message = self.loading_queue.get_nowait()
        except queue.Empty:
            return
        self.polling_timer.stop()
        logging.debug(f"Received message from loading thread: {message}")
        if isinstance(message, Exception):
            logging.error("Model loading failed.", exc_info=message)
            rumps.alert("Model Loading Failed", str(message))
            self.update_status("Error")
        elif message == "success":
            self.finish_setup()

    def load_models(self):
        try:
            logging.debug("MenuBarApp: load_models thread started.")
            self.client = openai.OpenAI()
            torch.set_num_threads(1)
            logging.debug("Loading Silero VAD model...")
            self.model_vad, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
            logging.debug("Loading Whisper model...")
            self.model_whisper = whisper.load_model(config_module.config.whisper.model)
            logging.debug("All models loaded successfully.")
            self.loading_queue.put("success")
        except Exception as e:
            self.loading_queue.put(e)

    def finish_setup(self):
        logging.debug("MenuBarApp: finish_setup triggered.")
        self.update_status("Ready")

    def update_status(self, text):
        logging.debug(f"Updating status to: {text}")
        self.title = f"⚫️ {text}"

    def toggle_monitoring(self, sender):
        if not self.model_whisper:
            rumps.alert("Models are still loading, please wait.")
            return
        self.is_monitoring = True
        self.title = "⚫️ Listening..."
        sender.title = "Stop Recording"
        self.start_monitoring()
        sender.set_callback(self.stop_monitoring)

    def stop_monitoring(self, sender):
        self.is_monitoring = False
        self.title = "⚫️ Ready"
        sender.title = "Start Recording"
        self.stop_event.set()
        sender.set_callback(self.toggle_monitoring)
        logging.info("Monitoring stopped.")

    def on_exit(self, _):
        logging.info("Exiting application, performing cleanup...")
        if self.is_monitoring:
            self.stop_event.set()
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logging.debug("Waiting for monitoring thread to join...")
            self.monitoring_thread.join()
            logging.debug("Monitoring thread joined.")
        rumps.quit_application()

    def start_monitoring(self):
        logging.debug("Attempting to start monitoring.")
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        self.stop_event.clear()
        self.monitoring_thread = threading.Thread(target=self.monitor_speech, args=(self.stop_event, self.update_status))
        self.monitoring_thread.start()
        logging.info("Monitoring started.")

    def process_audio_to_cursor(self, audio_frames, status_callback):
        logging.debug("Processing audio to cursor.")
        try:
            status_callback("Transcribing...")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_file:
                filename = tmp_audio_file.name
            audio_format = FORMAT_MAP[config_module.config.audio.format]
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(config_module.config.audio.channels)
                wf.setsampwidth(pyaudio.PyAudio().get_sample_size(audio_format))
                wf.setframerate(config_module.config.audio.sample_rate)
                wf.writeframes(b''.join(audio_frames))
            text = self.transcribe_audio(filename)
            if text.strip().lower() == config_module.config.stopword.lower():
                logging.info(f"Stopword '{config_module.config.stopword}' detected. Stopping recording.")
                rumps.Timer(lambda _: self.stop_monitoring(self.toggle_button), 0.1).start()
                return
            if text.strip():
                pyperclip.copy(text)
                self.paste_into_cursor()
        finally:
            if 'filename' in locals() and os.path.exists(filename):
                os.remove(filename)

    def monitor_speech(self, stop_event, status_callback):
        logging.debug("Monitor speech thread started.")
        p = pyaudio.PyAudio()
        audio_format = FORMAT_MAP[config_module.config.audio.format]
        stream = p.open(format=audio_format,
                        channels=config_module.config.audio.channels,
                        rate=config_module.config.audio.sample_rate,
                        input=True,
                        frames_per_buffer=config_module.config.audio.chunk_size)
        self.model_vad.reset_states()
        recorded_frames, is_speaking, silence_chunks = [], False, 0
        max_silence_chunks = int(config_module.config.vad.min_silence_duration_ms / 1000 * config_module.config.audio.sample_rate / config_module.config.audio.chunk_size)
        min_speech_chunks = int(config_module.config.vad.min_speech_duration_ms / 1000 * config_module.config.audio.sample_rate / config_module.config.audio.chunk_size)
        while not stop_event.is_set():
            data = stream.read(config_module.config.audio.chunk_size, exception_on_overflow=False)
            audio_float32 = np.frombuffer(data, dtype=np.int16).astype(np.float32) / INT16_TO_FLOAT_SCALE
            speech_prob = self.model_vad(torch.from_numpy(audio_float32), config_module.config.audio.sample_rate).item()
            if speech_prob > config_module.config.vad.threshold:
                if not is_speaking:
                    if len(recorded_frames) == 0: logging.debug("Speech detected.")
                    recorded_frames.append(data)
                    if len(recorded_frames) >= min_speech_chunks:
                        status_callback("Recording...")
                        is_speaking = True
                else:
                    recorded_frames.append(data)
                    silence_chunks = 0
            elif is_speaking:
                recorded_frames.append(data)
                silence_chunks += 1
                if silence_chunks >= max_silence_chunks:
                    logging.debug("Silence detected, processing audio.")
                    self.process_audio_to_cursor(recorded_frames, status_callback)
                    recorded_frames, is_speaking = [], False
                    status_callback("Listening...")
            else:
                recorded_frames = []
        stream.stop_stream()
        stream.close()
        p.terminate()
        logging.debug("Monitor speech thread finished.")

    def transcribe_audio(self, filename):
        logging.debug(f"Transcribing audio file: {filename}")
        result = self.model_whisper.transcribe(filename)
        logging.debug(f"Transcription result: {result['text']}")
        return result['text']

    def paste_into_cursor(self):
        logging.debug("Pasting into cursor.")
        delay = config_module.config.paste_delay
        applescript = f'''
        tell application "Cursor"
            activate
            delay {delay}
            tell application "System Events"
                keystroke "p" using {{command down, shift down}}
                delay {delay}
                keystroke "Focus Chat"
                delay {delay}
                keystroke return
                delay {delay}
                keystroke "v" using command down
            end tell
        end tell
        '''
        os.system(f"osascript -e '{applescript}'")

def run_app():
    warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
    app = MenuBarApp()
    return app 