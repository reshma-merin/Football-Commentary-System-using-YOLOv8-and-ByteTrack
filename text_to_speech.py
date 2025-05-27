from gtts import gTTS
import os

class TextToSpeech:
    def __init__(self, language='en', slow=False):
        self.language = language
        self.slow = slow

    def text_to_audio(self, text, filename):
        """
        Convert text to speech and save it as an MP3 file.

        Args:
            text (str): The commentary text.
            filename (str): The output MP3 filename.
        """
        if not text:
            print("No text provided for TTS.")
            return

        tts = gTTS(text=text, lang=self.language, slow=self.slow)
        tts.save(filename)
        print(f"Saved audio: {filename}")

# Example usage
if __name__ == "__main__":
    tts = TextToSpeech()
    tts.text_to_audio("Goal by player 7!", "sample_goal_commentary.mp3")



