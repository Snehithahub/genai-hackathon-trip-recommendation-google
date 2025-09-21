import streamlit as st
import os
import re
import json
from datetime import datetime, timedelta
import requests
from langchain.chat_models import init_chat_model
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
from serpapi import GoogleSearch

st.set_page_config(page_title="Trip Planner", layout="wide")

google_api_key = "AIzaSyDMQEc_Q0ZxGg5PxiV6odhQGNrHh86Pgww"
serpapi_key = "e18fb914caef7c7f2ea422d8ba76bea0f4d7a9f688c0f08601fdd22021d6227d"
openweather_key = "a7a970d978eb64fed9934bb64921389f"

if google_api_key == "YOUR_GOOGLE_GEMINI_API_KEY_HERE" or \
   serpapi_key == "YOUR_SERPAPI_KEY_HERE" or \
   openweather_key == "YOUR_OPENWEATHERMAP_API_KEY_HERE":
    st.error("Please replace the placeholder API keys in the code with your actual keys.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = google_api_key
os.environ["SERPAPI_KEY"] = serpapi_key


model = init_chat_model("gemini-1.5-flash", model_provider="google_genai")

def find_trip_details(user_input, model):
    prompt_template = PromptTemplate(
        input_variables=["user_input"],
        template="""
        Extract the following details from the user input:
        - destination
        - start_date
        - start_time
        - total_time (days, hours, minutes)
        - number_of_people
        - estimated_cost

        Respond strictly in JSON format. Use null for missing details.
        User input: "{user_input}"
        """
    )
    final_prompt = prompt_template.format(user_input=user_input)
    response = model.invoke(final_prompt)
    raw_response = response.content
    cleaned = re.sub(r"^```(json)?|```$", "", raw_response.strip(), flags=re.MULTILINE).strip()
    try:
        details = json.loads(cleaned)
    except Exception as e:
        details = {"error": "Invalid JSON from model", "raw": raw_response}
    return details

def fill_trip_times(details):
    now = datetime.now()
    if details.get("start_date"):
        try:
            start_date = datetime.strptime(details["start_date"], "%d-%m-%Y")
        except:
            start_date = now
    else:
        start_date = now

    if details.get("start_time"):
        try:
            t = datetime.strptime(details["start_time"], "%I:%M %p")
            start_datetime = datetime.combine(start_date.date(), t.time())
        except:
            start_datetime = datetime.combine(start_date.date(), now.time())
    else:
        start_datetime = datetime.combine(start_date.date(), now.time())

    details["start_time"] = start_datetime.strftime("%I:%M %p, %d-%m-%Y")

    days = 1
    if details.get("total_time"):
        m = re.search(r"(\d+)", str(details["total_time"]))
        if m:
            days = int(m.group(1))
    end_datetime = start_datetime + timedelta(days=days)
    details["end_time"] = end_datetime.replace(hour=23, minute=59).strftime("%I:%M %p, %d-%m-%Y")

    return details

def web_search(query: str) -> str:
    params = {"engine": "google", "q": query, "api_key": serpapi_key, "num": "10"}
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        info = [f"{r.get('title')}: {r.get('link')}" for r in results.get("organic_results", [])]
        return "\n".join(info)
    except Exception as e:
        return f"Error during web search: {e}"

def get_restaurants(city, num=5):
    query = f"top restaurants in {city}"
    search = GoogleSearch({"q": query, "location": city, "hl": "en", "gl": "in", "api_key": serpapi_key})
    results = search.get_dict()
    local_results = results.get("local_results")
    if not local_results or not isinstance(local_results, list):
        return []
    restaurants = [{
        "name": r.get("title"),
        "address": r.get("address"),
        "rating": r.get("rating"),
        "link": r.get("link")
    } for r in local_results[:num]]
    return restaurants

def get_weather(input_text: str) -> str:
    try:
        m_city = re.search(r"city:\s*([^,]+)", input_text)
        m_date = re.search(r"date:\s*(\d{2}-\d{2}-\d{4})", input_text)
        city = m_city.group(1).strip() if m_city else None
        date = m_date.group(1).strip() if m_date else None
        if not city or not date:
            return "Error: city or date missing"
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={openweather_key}&units=metric"
        resp = requests.get(url).json()
        if resp.get("cod") != "200":
            return f"Error fetching weather: {resp.get('message', 'City not found')}"
        for item in resp.get("list", []):
            dt_txt = item["dt_txt"].split()[0]
            if dt_txt == datetime.strptime(date, "%d-%m-%Y").strftime("%Y-%m-%d"):
                weather = item["weather"][0]["description"]
                temp = item["main"]["temp"]
                return f"{weather}, {temp}°C"
        return "Weather data not available"
    except Exception as e:
        return f"Error fetching weather: {e}"

search_tool = Tool(name="WebSearch", func=web_search, description="Search for attractions, restaurants, temples, activities, and food in a city")
weather_tool = Tool(name="Weather", func=get_weather, description="Get weather for a city on a specific date. Provide input like 'city: Tirupati, date: 20-09-2025'.")
restaurant_tool = Tool(name="Restaurants", func=get_restaurants, description="Get top restaurants in a city")

agent = initialize_agent(
    tools=[search_tool, weather_tool, restaurant_tool],
    llm=model,
    agent="zero-shot-react-description",
    handle_parsing_errors=True,
    verbose=True
)

st.markdown("<h1 style='text-align: center;'>Personalized Trip Planner with AI✈️</h1>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm your AI trip planner. Tell me about your trip (e.g., '3-day trip to Tirupati for 2 people')."})

if "trip_details" not in st.session_state:
    st.session_state.trip_details = {}

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Start planning your trip...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Processing your request..."):
        new_details = find_trip_details(user_input, model)

        for key, value in new_details.items():
            if value is not None and value != "" and key not in ["error", "raw"]:
                st.session_state.trip_details[key] = value

        st.session_state.trip_details = fill_trip_times(st.session_state.trip_details)

        missing_fields = []
        if not st.session_state.trip_details.get("destination"):
            missing_fields.append("destination")
        if not st.session_state.trip_details.get("number_of_people"):
            missing_fields.append("number of people")
        
        if missing_fields:
            response_text = f"I need a few more details to plan your trip. Please provide the {', '.join(missing_fields)}."
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.rerun()
        else:
            agent_prompt = f"""
            Plan a detailed trip itinerary based on the following trip details:
            Destination: {st.session_state.trip_details['destination']}
            Start Time: {st.session_state.trip_details['start_time']}
            End Time: {st.session_state.trip_details['end_time']}
            Number of People: {st.session_state.trip_details['number_of_people']}
            Estimated Cost: {st.session_state.trip_details.get('estimated_cost', 'Not provided')}

            For each day, fetch:
            - attractions, temples, activities using WebSearch tool
            - top restaurants using Restaurants tool
            - weather using Weather tool

            Output the itinerary using Markdown format with bold headings and bullet points.
            The final output should NOT be a JSON object, just clean, well-formatted text.
            """
            response_text = agent.run(agent_prompt)

            with st.chat_message("assistant"):
                st.markdown("Here is your generated trip itinerary:")
                st.markdown(response_text)

            st.session_state.messages.append({"role": "assistant", "content": response_text})
