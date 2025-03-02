import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from gtts import gTTS
import os
import speech_recognition as sr
from datasets import load_dataset
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

# Load Hinglish LLM
model_name = "Abhishekcr448/Tiny-Hinglish-Chat-21M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

# State Management
state = {
    "customer_name": None,
    "demo_date": None,
    "interview_progress": None,
    "payment_status": None
}


# Generate Hinglish text
def generate_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    output = model.generate(
        inputs['input_ids'],
        max_length=max_length,
        no_repeat_ngram_size=2,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        do_sample=True
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


# Text to speech
def speak_text(text):
    if text:
        tts = gTTS(text=text, lang="hi")
        # tts.save("response.mp3")
        os.system("start response.mp3")  # For Windows


# Cold Calling Scenarios

def handle_demo(user_input):
    prompt = f" '{user_input}'."
    return generate_text(prompt)


def handle_interview(user_input):
    prompt = f"'{user_input}'."
    return generate_text(prompt)


def handle_payment(user_input):
    prompt = f"'{user_input}'."
    return generate_text(prompt)


# Detect scenario
def detect_scenario(user_input):
    user_input = user_input.lower()
    if "demo" in user_input or "schedule" in user_input:
        return "demo"
    elif "interview" in user_input or "candidate" in user_input:
        return "interview"
    elif "payment" in user_input or "order" in user_input:
        return "payment"
    else:
        return "general"


# Voice input function
def listen_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n🎙️ Bolna shuru karein... (say 'exit' to stop)")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio, language="en-IN")
        print(f"User (voice): {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, samajh nahi aaya. Please repeat.")
        return None
    except sr.RequestError as e:
        print(f"Error: {e}")
        return None


# Main flow
def main():
    print("Hinglish Cold Calling Agent Ready with Voice Input! (Say 'exit' to stop)\n")
    while True:
        user_input = listen_voice()
        if user_input is None:
            continue
        if user_input.lower() == "exit":
            break

        scenario = detect_scenario(user_input)

        if scenario == "demo":
            hinglish_text = handle_demo(user_input)
        elif scenario == "interview":
            hinglish_text = handle_interview(user_input)
        elif scenario == "payment":
            hinglish_text = handle_payment(user_input)
        else:
            hinglish_text = "Sorry, main samajh nahi paya. Kya aap repeat karoge?"

        print(f"Agent: {hinglish_text}")
        speak_text(hinglish_text)


if __name__ == "__main__":
    main()
