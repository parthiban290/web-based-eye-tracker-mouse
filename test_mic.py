import speech_recognition as sr

print("=== Microphone Test ===")
print("")

# Show all microphones
mics = sr.Microphone.list_microphone_names()
print("Available microphones:")
for i, name in enumerate(mics):
    print(f"  [{i}] {name}")

print("")

# Test recording
r = sr.Recognizer()
m = sr.Microphone()

print("Adjusting for background noise...")
with m as source:
    r.adjust_for_ambient_noise(source, duration=1)

print("")
print(">>> SPEAK NOW! You have 5 seconds <<<")
print("")

with m as source:
    try:
        audio = r.listen(source, timeout=5)
        print("Processing your speech...")
        text = r.recognize_google(audio)
        print("")
        print(f"You said: {text}")
        print("")
        print("SUCCESS! Your microphone works!")
    except sr.WaitTimeoutError:
        print("ERROR: No speech detected in 5 seconds")
    except sr.UnknownValueError:
        print("ERROR: Could not understand what you said")
    except sr.RequestError as e:
        print(f"ERROR: Google API error - {e}")
    except Exception as e:
        print(f"ERROR: {e}")