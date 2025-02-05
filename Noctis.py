# import sys
# import speech_recognition as sr
# import pyttsx3
# from BERT.main import recognize_intent  # Ensure this path is correct
#
# def STT():
#     recognizer = sr.Recognizer()
#     with sr.Microphone() as mic:
#         print('Say something ...')
#         recognizer.adjust_for_ambient_noise(mic, 1)
#         audio = recognizer.listen(mic)
#         print('Processing speech')
#
#     try:
#         text = recognizer.recognize_google(audio)
#         if text.lower() == 'goodbye' or text.lower() == 'bye':
#             sys.exit()
#         else:
#             intent = recognize_intent(text)
#             print(f'Recognized Intent: {intent}')
#             return text, intent
#
#     except sr.UnknownValueError:
#         print("Sorry, I could not understand the audio.")
#         return None
#     except sr.RequestError as e:
#         print("Could not request results; {0}".format(e))
#         return None
#
#
# def TTS(text):
#     engine = pyttsx3.init()
#     engine.say(text)
#     engine.runAndWait()
#
# def main():
#     # Convert speech to text and recognize intent
#     result = STT()
#
#     # If we got some text, convert it back to speech
#     if result:
#         text, intent = result
#         print("Speaking out the recognized text and intent...")
#         TTS(f"You said: '{text}'. Detected intent is: '{intent}'.")
#     else:
#         print("No valid speech input to process.")
#
# if __name__ == "__main__":
#     while True:
#         main()

from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "./BERT/Intent-Recognition/trained_model"  # Ensure this path is correct (absolute if needed)

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
bert_model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)

