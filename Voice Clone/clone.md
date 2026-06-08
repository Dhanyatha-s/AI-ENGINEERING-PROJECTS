# CLONE YOUR VOICE USING XTTS_v2 MODEL
<img width="500" height="481" alt="image" src="https://github.com/user-attachments/assets/bf50633d-6d10-409e-ac7b-420bc980fc9f" />


#### Requirements
- Original_voice.wav [!Note: The Voice Must be in .wav format if not convert it using pydub AudioSegment]
- device -> cuda
- Package/librarie - [coqui-tts, transformers>=4.57,<5, mecab-python3, unidic-lite, pydub]
- Model -> XTTS_v2 (tts_model/multilingual/multi-dataset/xtts_v2)


#### Steps
1. Store your Orginal File in Drive [.mp3 or .wav]
2. Open Google Colab
  - Change Runtime to T4 CUDA
3. Install Packages
  - ```
    !pip install -q "transformers>=4.57,<5" coqui-tts pydub mecab-python3 unidic-lite --upgrade
    ```
4a. If file is in .mp3 convert to .wav using pydub
```
from google.colab import files
from pydub import AudioSegment

# Load file from files
files = "your-file-name.mp3"

convert_wav = AudioSegment.from_mp3(files)
convert_wav = conver_wav.set_frame_rate(2250).set_channel = 1)

wav_file = "converted_voice.wav"
convert_wav.export(wav_file, format='wav')
print("Conversion is successful")
```
4b. Clone the Voice
```
# Load file, model and Clone
from TTS.api import TTS

#Initialize the model
tts = TTS(model_name = "tts_model/multilingual/multi-dataset/xtts_v2".gpu=True)

# Set target text for text-to-speech

traget_text = "The audio conversion is complete. My target speech file was successfully changed from an MP3 to a WAV layout, and now the AI model can mirror my exact voice flawlessly."
output_filename = "cloned_voice.wav"

# Generate voice
tts.tts_to_file(
  text = target_text,
  speaker_wav = "your-original_voice.wav"
  language = 'en',
  file_path = output_filename)

print("successfully Cloned your Voice")

```
References - 
Cloned Voice -> [cloned_voice.wav](https://github.com/user-attachments/files/28712151/cloned_voice.wav)



