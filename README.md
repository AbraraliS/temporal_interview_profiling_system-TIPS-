# 🎯 Temporal Interview Profiling System (TIPS)

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge)
![Whisper](https://img.shields.io/badge/Whisper-ASR-6A5ACD?style=for-the-badge)
![Local LLM](https://img.shields.io/badge/LLM-Local%20Inference-00b894?style=for-the-badge)
![License](https://img.shields.io/badge/License-Custom%20Proprietary-red?style=for-the-badge)
![Platform](https://img.shields.io/badge/Platform-Linux%20|%20Windows%20|%20Mac-black?style=for-the-badge)

---

## 📌 Overview

This project is a **local, Profiling System**.
It processes **video + audio + transcript** in real-time to understand:

* 👀 **Facial emotions**
* 🎤 **Voice-based cues & prosody**
* 🧠 **Speech-to-text transcription**
* 🗣️ **Question → Answer mapping**
* 🤖 **LLM summaries of candidate performance**
* 📊 **Realtime & final dashboards**

All processing runs **locally**, without cloud dependencies.

---

## 🧩 What This System Does

✔ Detects faces in real-time
✔ Predicts visual emotions (happy, neutral, surprise, confusion, etc.)
✔ Extracts audio features (pitch, energy, fillers, speaking style)
✔ Converts voice → text using offline Whisper models
✔ Separates interviewer & candidate (optional diarization)
✔ Builds a question–answer structure from conversation
✔ Sends compact segments to a local LLM for:

* Short realtime feedback
* A complete final interview report
  ✔ Generates dashboards for monitoring and reviewing each session

---

## 🛠️ Core Components

* **Visual Emotion Pipeline** → Face Detection → Crop → Emotion Model
* **Audio Processing** → VAD → ASR → Prosody → Diarization
* **Q/A Mapper** → Detects interviewer questions & candidate answers
* **LLM Summarizer** → Human-like insights from structured data
* **Dashboard** → Realtime + Post-Interview Analysis

---

## 🚀 Goals

* Fully local AI-powered interview analysis
* Real-time insights for interviewers
* Fair, auditable, transparent evaluation tools
* Easy to extend with new models & scoring logic

