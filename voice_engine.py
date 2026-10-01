import pyautogui
import time
import threading
import speech_recognition as sr

pyautogui.FAILSAFE = False


class VoiceTypingEngine:
    def __init__(self, language='en-US'):
        self.language = language
        self.running = False
        self.active = True
        self.typed_count = 0
        self.last_text = ""

        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 350
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.5
        self.recognizer.phrase_threshold = 0.2
        self.recognizer.non_speaking_duration = 0.3

        mics = sr.Microphone.list_microphone_names()
        print(f"[Voice] Available mics: {mics}")

        self.microphone = sr.Microphone()
        self.initialized = True
        print(f"[Voice] Ready (lang={language})")

    def run(self):
        self.running = True

        try:
            with self.microphone as source:
                print("[Voice] Adjusting for noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("[Voice] LISTENING STARTED!")
        except Exception as e:
            print(f"[Voice] Mic error: {e}")
            self.running = False
            return

        while self.running:
            if not self.active:
                time.sleep(0.2)
                continue

            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=10)

                t = threading.Thread(target=self._process, args=(audio,), daemon=True)
                t.start()

            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                if self.running:
                    print(f"[Voice] Error: {e}")
                    time.sleep(0.3)

        print("[Voice] Stopped")

    def _process(self, audio):
        try:
            text = self.recognizer.recognize_google(audio, language=self.language)

            if text and text.strip():
                clean = text.strip()
                print(f"[Voice] Heard: '{clean}'")

                try:
                    import pyperclip
                    pyperclip.copy(clean + ' ')
                    time.sleep(0.03)
                    pyautogui.hotkey('ctrl', 'v')
                except ImportError:
                    pyautogui.typewrite(clean + ' ', interval=0.02)

                self.typed_count += 1
                self.last_text = clean
                print(f"[Voice] Typed! (total: {self.typed_count})")

        except sr.UnknownValueError:
            pass
        except sr.RequestError as e:
            print(f"[Voice] API error: {e}")
        except Exception as e:
            print(f"[Voice] Process error: {e}")

    def stop(self):
        self.running = False
        self.active = False

    def pause(self):
        self.active = False

    def resume(self):
        self.active = True