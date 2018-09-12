from googletrans import Translator

from prediction import pred
import os

translator = Translator()
translations = translator.translate([pred], dest='bn')
for translation in translations:
    print(translation.origin, ' -> ', translation.text)

from gtts import gTTS
from tempfile import TemporaryFile
text = pred #translation.text
targetLanguage = 'bn'
tts = gTTS(text, targetLanguage)
tts.save("9.mp3")
os.system('9.mp3')