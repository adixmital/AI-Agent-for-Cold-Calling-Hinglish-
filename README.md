# 🚀 Hinglish Cold Calling AI Agent

### 📜 Project Description

This project is an **AI-powered Hinglish Cold Calling Agent** capable of handling real-world sales scenarios like:

- 📅 Demo scheduling
- 🧑‍💼 Interview responses
- 💰 Payment follow-ups

The agent listens to **voice inputs**, understands the intent, generates appropriate Hinglish responses, and replies back using **Text-to-Speech (TTS)**, making it ideal for customer interaction automation in Indian contexts.

---

## ⚙️ Environment Setup & Running the Agent

### ✅ 1. Clone the repository

```bash
git clone https://github.com/adixmital/AI-Agent-for-Cold-Calling-Hinglish-.git
cd AI-Agent-for-Cold-Calling-Hinglish-

```

### ✅ 2. Install dependencies

```bash
pip install -r requirements.txt
```

### ✅ 3. Required packages include:

```bash
transformers
torch
gTTS
SpeechRecognition
PyAudio
GitPython
PyGithub
datasets
```

### ✅ 4. Run the Agent

```bash
python agent.py
```

Then interact with the agent using your microphone in Hinglish!

---

## 🧠 Models & Datasets Used

| Component                    | Model/Dataset                                           | Purpose                        |
| ---------------------------- | ------------------------------------------------------- | ------------------------------ |
| Hinglish LLM                 | `Abhishekcr448/Tiny-Hinglish-Chat-21M`                  | Generating Hinglish responses  |
| Text-to-Speech (TTS)         | `microsoft/speecht5_tts` + `microsoft/speecht5_hifigan` | Convert text to natural speech |
| Voice Embeddings             | `Matthijs/cmu-arctic-xvectors`                          | Speaker identity in TTS        |
| Dataset (optional fine-tune) | Custom `hinglish_cold_calls.jsonl`                      | Task-specific training         |

---

## 🏗️ Agent Architecture

```
Voice Input 🎙️  
     ↓  
Speech Recognition (English) 🗣️  
     ↓  
Scenario Detection (Demo, Interview, Payment) 🧠  
     ↓  
Prompt Engineering + Hinglish LLM 🔄  
     ↓  
Generated Response 📝  
     ↓  
Text-to-Speech (Hinglish Accent) 🔊  
     ↓  
Audio Output 🔈  
```

### 🔑 Key Components

- **Scenario Detection**: Determines the conversation type (demo, interview, payment).
- **LLM Text Generation**: Produces natural Hinglish text based on user input.
- **Voice Input**: Captures speech via microphone.
- **Speech Output**: Plays the generated Hinglish response.

---

## 🎥 Demonstration Video

Watch the full demo
🔗 https://drive.google.com/drive/folders/1-96RSg7T_zP8xXC8NiW8Oieeifv2FUMn?usp=drive_link

---

## ✅ Project Completion Status

| Feature                                        | Status            |
| ---------------------------------------------- | ----------------- |
| Voice Input (Speech-to-Text)                   | ✅ Completed       |
| Hinglish Text Generation                       | ✅ Completed       |
| Text-to-Speech (TTS) in Hinglish Accent        | ✅ Completed       |
| Cold Call Scenarios (Demo, Interview, Payment) | ✅ Completed       |
| Hinglish Dataset Fine-tuning                   | ⚠️ Partially Done |
| Advanced Voice Emotion Handling                | ❌ Not Implemented |

---

## 🙏 Acknowledgements

- Hugging Face 🤗
- Microsoft SpeechT5 🗣️
- Google Text-to-Speech (gTTS) 🔊

