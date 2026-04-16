import pickle
import numpy as np
import os

from dotenv import load_dotenv

from langchain.tools import tool
from langchain_classic.agents import create_tool_calling_agent
from langchain_classic.agents import AgentExecutor

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate



from weather import get_weather_info

# ==========================
# Load Environment
# ==========================

load_dotenv()

# ==========================
# Load ML Model
# ==========================

with open("models/fertilizer.pkl", "rb") as f:
    model = pickle.load(f)

# ==========================
# Config
# ==========================

api_key = "52d129c124aaa073f4208583f3299b19"
Global_location = "Kolhapur"


# ==========================
# MAPPINGS
# ==========================

crop_mapping = {
    "barley": 0,
    "cotton": 1,
    "ground nuts": 2,
    "maize": 3,
    "millets": 4,
    "oil seeds": 5,
    "paddy": 6,
    "pulses": 7,
    "sugarcane": 8,
    "tobacco": 9,
    "wheat": 10
}

soil_mapping = {
    "black": 0,
    "clayey": 1,
    "loamy": 2,
    "red": 3,
    "sandy": 4
}

# ==========================
# TOOLS
# ==========================

@tool
def weather_tool():
    """Get current weather information."""

    data = get_weather_info(
        Global_location,
        api_key
    )

    temp = data['main']['temp']
    humidity = data['main']['humidity']
    rain = data.get('rain', {}).get('6h', 0)

    return {
        "temperature": temp,
        "humidity": humidity,
        "rain": rain
    }


@tool
def advice_tool():
    """Give farming advice based on weather."""

    data = get_weather_info(
        Global_location,
        api_key
    )

    temp = data['main']['temp']

    if temp < 10:
        return "Protect seedlings from cold."
    elif temp < 20:
        return "Suitable for cool-season crops."
    elif temp < 30:
        return "Ideal weather conditions."
    else:
        return "Watch for heat stress."

@tool
def fertilizer_tool(crop: str, soil: str):
    """Recommend fertilizer based on crop and soil."""

    crop = crop.lower()
    soil = soil.lower()

    if crop not in crop_mapping:
        return "Unknown crop."

    if soil not in soil_mapping:
        return "Unknown soil."

    weather = get_weather_info(
        Global_location,
        api_key
    )

    temp = weather['main']['temp']
    humid = weather['main']['humidity']

    crop_id = crop_mapping[crop]
    soil_id = soil_mapping[soil]

    data = [temp, humid, soil_id, crop_id]

    prediction = model.predict([data])

    return prediction[0]

# ==========================
# LLM
# ==========================

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.7
)

# ==========================
# AGENT
# ==========================

tools = [
    weather_tool,
    advice_tool,
    fertilizer_tool
]

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful agricultural assistant. "
        "Use tools whenever required."
    ),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(
    llm,
    tools,
    prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=3,
    max_execution_time=15
)

# ==========================
# RUN FUNCTION
# ==========================

def run_agent(query):

    response = agent_executor.invoke({
        "input": query
    })

    return response["output"]