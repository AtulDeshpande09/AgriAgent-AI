# AgriAgent-AI

An agent-based AI system that integrates real-time weather data, rule-based advisory logic, and machine learning-based fertilizer prediction using LangChain and Groq LLM API.

---

## Overview

AgriAgent-LangChain is a multi-tool intelligent agent designed to assist agricultural decision-making. The system combines external APIs, rule-based logic, and machine learning models into a unified agent framework.

The agent dynamically selects tools based on user queries and generates context-aware responses.

---

## Features

* 🌦️ Weather Retrieval Tool
  Fetches real-time weather data using OpenWeather API.

* 📊 Weather-Based Advisory Tool
  Generates recommendations based on temperature and humidity conditions.

* 🌱 Fertilizer Recommendation Tool
  Uses a trained machine learning model (.pkl file) to suggest fertilizer types.

* 🤖 Agent-Based Decision System
  Built using LangChain agent architecture.

* 🌐 Flask Web Interface
  Simple UI for user interaction.

* ⚡ Groq LLM Integration
  Uses Groq-powered language model for reasoning and orchestration.

---

## System Architecture

User Input → Agent → Tool Selection → Tool Execution → Response Generation

### Available Tools:

1. **Weather Tool**

   * Retrieves real-time weather information.
   * Uses OpenWeather API.

2. **Advisory Tool**

   * Generates suggestions based on weather conditions.
   * Uses rule-based logic.

3. **Fertilizer Prediction Tool**

   * Predicts fertilizer recommendations.
   * Uses pre-trained machine learning model.

---

## Tech Stack

* Python
* Flask
* LangChain
* Groq API
* OpenWeather API
* Scikit-learn
* HTML (UI)

---


## Environment Setup

Create a `.env` file and add:

GROQ_API_KEY=your_key_here /
or 
```bash
export GROQ_API_KEY="yourAPIkey"
```

---

## Run the Application

Run Flask app:

python app.py

Open browser:

http://127.0.0.1:5000/assistant


