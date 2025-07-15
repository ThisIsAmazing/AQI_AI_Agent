import streamlit as st
import os
import requests
from dotenv import load_dotenv
from typing import Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime
from pydantic import BaseModel, Field
import pandas as pd
from geopy.geocoders import Nominatim
from datetime import datetime
from prophet import Prophet
import matplotlib.pyplot as plt
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import ChatLiteLLM
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from crewai import Agent, Task, Crew, Process

load_dotenv() 
# -------------------------------------------------------------------
# 1. Define the Pydantic models for schema validation
# -------------------------------------------------------------------

class AQIResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    status: str
    expiresAt: Union[str, datetime]

class DailyAQIMeasurement(BaseModel):
    date: str = Field(description="Date of the measurement in YYYY-MM-DD format")
    aqi: float = Field(description="Air Quality Index")
    pollutant: float = Field(description="Primary pollutant in the air")

class ExtractSchema(BaseModel):
    measurements: list[DailyAQIMeasurement]

# -------------------------------------------------------------------
# 2. Define a simple dataclass to hold user inputs
# -------------------------------------------------------------------

@dataclass
class UserInput:
    city: str
    country: str
    medical_conditions: Optional[str]
    planned_activity: str

# -------------------------------------------------------------------
# 3. AQIAnalyzer: 
# -------------------------------------------------------------------

class AQIAnalyzer:

    def _format_url(self) -> str:
        """
        Format the target URL based on location, handling cases with and without state.
        """
        return f"https://airquality.googleapis.com/v1/history:lookup?key={os.getenv('API_KEY')}"
    
    def get_lat_lon_geopy(self, address):
        geolocator = Nominatim(user_agent="aqi checker")
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
        else:
            raise Exception("Location not found")

    def fetch_aqi_data(self, city: str, country: str) -> Tuple[Dict[Any, Any], str]:
        """
        Fetch AQI data from the website using Firecrawl. Returns a tuple:
          (data_dict, info_message)
        On failure, returns default zeroed metrics + an error message.
        """
        try:
            url = self._format_url()
            address = f"{city}, {country}"
            latitude, longitude = self.get_lat_lon_geopy(address)
            headers = {
                "Content-Type": "application/json"
            }
            body = {
                "hours": 168,
                "pageSize": 168,
                "pageToken":"",
                "location": {
                    "latitude": float(latitude),
                    "longitude": float(longitude)
                }
            }
            response = requests.post(url, headers=headers, json=body)

            if response.status_code == 200:
                data = response.json()
                measurements = []
                for hour_info in data["hoursInfo"]:
                    date_time = hour_info["dateTime"]
                    aqi = hour_info.get("indexes", [{}])[0].get("aqi", None) if "indexes" in hour_info and hour_info["indexes"] else None
                    pollutant = hour_info.get("indexes", [{}])[0].get("dominantPollutant", None) if "indexes" in hour_info and hour_info["indexes"] else None
                    if aqi is not None:
                        aqi = round(aqi, 1)
                    dt = datetime.strptime(date_time, "%Y-%m-%dT%H:%M:%SZ")
                    readable = dt.strftime("%B %d, %Y %I:%M %p UTC")

                    measurements.append({
                        "date": readable,
                        "aqi": aqi,
                        "pollutant": pollutant,
                    })

                data = {
                    "measurements": measurements
                }

                return data, "AQI data fetched successfully"
            else:
                raise Exception(f"Failed to fetch data: {response.status_code} - {response.text}")
        except Exception as e:
            return -1, -1

# -------------------------------------------------------------------
# 4. HealthRecommendationAgent placeholder
# -------------------------------------------------------------------

class HealthRecommendationAgent:

    def get_recommendations(
        self, current_aqi: str, user_input: UserInput
    ) -> str:
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            max_completion_tokens=None,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=1
        )

        user_input_str = (
            f"City: {user_input.city}, Country: {user_input.country}\n"
            f"Medical Conditions: {user_input.medical_conditions if user_input.medical_conditions else 'None'}\n"
            f"Planned Activity: {user_input.planned_activity}"
        )

        template = PromptTemplate(
            input_variables=["current_aqi", "user_input"],
            template="""
            Using the current AQI of {current_aqi} and the user's profile:
            {user_input}
            Provide personalized health recommendations for the user.

            """,
        )

        parser = StrOutputParser()

        chain = template | model | parser

        result = chain.invoke({
            "current_aqi": current_aqi,
            "user_input": user_input_str
        })

        return result
# -------------------------------------------------------------------
# 5. Main analysis function
# -------------------------------------------------------------------

def analyze_conditions(
    city: str,
    country: str,
    medical_conditions: Optional[str],
    planned_activity: str,
) -> Tuple[pd.DataFrame, str, str]:
    try:
        aqi_analyzer = AQIAnalyzer()
        health_agent = HealthRecommendationAgent()
        user_input = UserInput(
            city=city.strip(),
            country=country.strip(),
            medical_conditions=medical_conditions.strip() if medical_conditions else None,
            planned_activity=planned_activity.strip(),
        )

        aqi_data, info_msg = aqi_analyzer.fetch_aqi_data(
            city=user_input.city,
            country=user_input.country,
        )
        
        if aqi_data == -1 and info_msg == -1:
            return pd.DataFrame(), "Unable to retrieve AQI data", f"No AQI data found for {city}, {country}. Please check the city and country names and try again."

        if "measurements" in aqi_data and isinstance(aqi_data["measurements"], list):
            df = pd.DataFrame(aqi_data["measurements"])
            df.rename(columns={
                "aqi": "Air Quality Index (AQI)",
                "pollutant": "Primary Pollutant",
            }, inplace=True)
        
        current_aqi = df['Air Quality Index (AQI)'].iloc[167]
        recommendations = health_agent.get_recommendations(current_aqi, user_input)
        df = df[::-1].reset_index(drop=True)
        return df, recommendations, info_msg

    except Exception as e:
        error_msg = f"Error occurred during analysis: {str(e)}"
        return pd.DataFrame(), "Analysis failed", error_msg

# -------------------------------------------------------------------
# 6. Prediction function
# -------------------------------------------------------------------

def makePrediction(df :pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=False,
        seasonality_mode='additive'
    )

    df.rename(columns={'date': 'ds', 'Air Quality Index (AQI)': 'y'}, inplace=True)

    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)

    model.fit(df)

    future = model.make_future_dataframe(periods=24, freq='H') 
    forecast = model.predict(future)

    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(24)

    return forecast

# -------------------------------------------------------------------
# 7. AI Agents
# -------------------------------------------------------------------   
    
@tool
def search_web_tool(query: str):
    search_tool = DuckDuckGoSearchResults(num_results=10, verbose=True)
    return search_tool.run(query)

mask_finder = Agent(
    role="Mask Finder",
    goal="Find the best mask for air quality protection",
    backstory="""You are an expert in air quality protection and can find the best masks for users based on their needs.""",
    tools=[search_web_tool],
    max_iter=5,
    llm=ChatLiteLLM(model="gemini/gemini-2.0-flash", temperature=1, api_key=os.getenv("GOOGLE_API_KEY")),
    allow_delegation=False
)

mask_task = Task(
    description="Find the best mask for air quality protection",
    expected_output="A list of recommended masks with links to purchase them.",
    agent=mask_finder
)

crew = Crew(
    agents=[mask_finder],
    tasks=[mask_task],
    process = Process.sequential,
    full_output=True,
    share_crew=False,
)

def get_mask_recommendations():
    result = crew.kickoff()
    return result

# -------------------------------------------------------------------
# 8. Streamlit interface
# -------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="AQI Health Advisor",
        page_icon="🌬️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #2c3e50;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #34495e;
            margin-bottom: 1rem;
        }
        .info-text {
            font-size: 1rem;
            color: #7f8c8d;
        }
        .warning {
            background-color: #fff3cd;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 0.5rem solid #ffc107;
            margin: 1rem 0;
        }
        .card {
            background-color: green;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        .mask-card {
            background-color: #3498db;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
            border-left: 0.5rem solid #3498db;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 class='main-header'>AQI Health Advisor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='info-text'>Get personalized air quality insights and health recommendations based on your location and planned activities.</p>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("<h2 class='sub-header'>Your Information</h2>", unsafe_allow_html=True)
        
        with st.form("user_input_form"):
            city = st.text_input("City", placeholder="E.g. Mumbai")
            country = st.text_input("Country", placeholder="E.g. India")
            medical_conditions = st.text_area("Medical Conditions (if any)", 
                                             placeholder="E.g., asthma, allergies, etc.")
            planned_activity = st.text_input("Planned Activity", 
                                           placeholder="E.g., jogging, hiking, etc.",)
            
            submit_button = st.form_submit_button("Get AQI Analysis")
    
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    if 'mask_recommendations' not in st.session_state:
        st.session_state.mask_recommendations = None
    
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = 0 

    if submit_button:
        with st.spinner("Analyzing air quality data..."):
            st.session_state.mask_recommendations = None
            
            try:
                df, recommendations, info_msg = analyze_conditions(
                    city=city,
                    country=country,
                    medical_conditions=medical_conditions,
                    planned_activity=planned_activity,
                )
                
                if "No AQI data found for" in info_msg:
                    st.error(info_msg)
                    st.warning("Please try a different city or check the spelling of your location.")
                    st.session_state.results = None
                else:
                    st.session_state.results = {
                        'df': df.copy(),
                        'recommendations': recommendations,
                        'info_msg': info_msg
                    }
                    
                    st.success("Analysis completed successfully!")
            except Exception as e:
                st.error(f"Error occurred: {str(e)}")
                st.warning("Please try a different city or check your network connection.")
                st.session_state.results = None
    
    if st.session_state.results:
        results = st.session_state.results
        df = results['df']
        recommendations = results['recommendations']
        info_msg = results['info_msg']
        
        tab1, tab2, tab3, tab4 = st.tabs(["AQI Information", "Health Recommendations", "Mask Recommendations", "Historical AQI Data"])
        
        with tab1:
            st.header("AQI Information")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<div class='card'><h3>Location</h3><p>{city}, {country}</p></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='card'><h3>Activity</h3><p>{planned_activity}</p></div>", unsafe_allow_html=True)
            
            if not df.empty:
                prediction_df = df.copy()
                
                try:
                    forecast = makePrediction(prediction_df)
                    forecast_table = forecast.tail(24)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                    forecast_table['ds'] = forecast_table['ds'].dt.strftime('%Y-%m-%d %H:%M')
                    
                    forecast_table = forecast_table.rename(columns={
                        'ds': 'Date & Time(UTC)',
                        'yhat': 'Predicted AQI',
                        'yhat_lower': 'Low Estimate',
                        'yhat_upper': 'High Estimate'
                    })
                    
                    for col in ['Predicted AQI', 'Low Estimate', 'High Estimate']:
                        forecast_table[col] = forecast_table[col].round(1)
                    
                    def color_aqi(val):
                        if pd.isna(val):
                            return ''
                        try:
                            val = float(val)
                        except:
                            return ''
                        if 80 <= val <= 100:
                            color = '#009E3A'
                        elif 60 <= val < 80:
                            color = '#84CF33'
                        elif 40 <= val < 60:
                            color = '#FFFF00'
                        elif 20 <= val < 40:
                            color = '#FF8C00'
                        elif 1 <= val < 20:
                            color = '#FF0000'
                        elif val == 0:
                            color = '#800000'
                        else:
                            color = ''
                        return f'color: {color}; font-weight: bold;'
                    
                    st.markdown("<h3>24-Hour AQI Forecast</h3>", unsafe_allow_html=True)
                    st.dataframe(
                        forecast_table.style.applymap(color_aqi, subset=['Predicted AQI', 'Low Estimate', 'High Estimate']),
                        use_container_width=True,
                        height=500,
                        hide_index=True
                    )
                    
                    st.markdown("<p class='info-text'>The table shows predicted Air Quality Index (AQI) for the next 24 hours. Higher values (green) indicate better air quality, while lower values (red) indicate worse air quality. The values in the chart use the Universal Air Quality Index(UAQI). </p>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error generating prediction: {str(e)}")
            else:
                st.warning("Insufficient data to make predictions.")
        
        with tab2:
            st.header("Health Recommendations")
            
            st.markdown(f"<div class='card'>{recommendations.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
            if medical_conditions:
                st.markdown("<h3>Medical Condition Considerations</h3>", unsafe_allow_html=True)
                
                condition_advice = {
                    "asthma": """
                        <div class='card'>
                            <h4>Asthma Management</h4>
                            <ul>
                                <li>Keep your rescue inhaler with you at all times</li>
                                <li>Consider using a preventative inhaler before outdoor activities</li>
                                <li>Monitor your peak flow readings more frequently</li>
                                <li>Stay hydrated to keep airways moist</li>
                            </ul>
                        </div>
                    """,
                    "allergies": """
                        <div class='card'>
                            <h4>Allergy Management</h4>
                            <ul>
                                <li>Consider taking antihistamines before outdoor exposure</li>
                                <li>Wear sunglasses to protect eyes from irritants</li>
                                <li>Shower after outdoor activities to remove allergens</li>
                                <li>Consider a saline nasal rinse to clear irritants</li>
                            </ul>
                        </div>
                    """,
                    "copd": """
                        <div class='card'>
                            <h4>COPD Management</h4>
                            <ul>
                                <li>Limit outdoor exposure when AQI is elevated</li>
                                <li>Use prescribed oxygen as directed</li>
                                <li>Practice pursed-lip breathing techniques</li>
                                <li>Stay current with vaccinations</li>
                            </ul>
                        </div>
                    """
                }
                
                displayed_advice = False
                for condition, advice in condition_advice.items():
                    if condition in medical_conditions.lower():
                        st.markdown(advice, unsafe_allow_html=True)
                        displayed_advice = True
                
                if not displayed_advice:
                    st.markdown("""
                        <div class='card'>
                            <h4>General Health Considerations</h4>
                            <p>With your mentioned medical conditions, consider:</p>
                            <ul>
                                <li>Carrying all necessary medications</li>
                                <li>Informing companions about your condition</li>
                                <li>Taking more frequent breaks during activities</li>
                                <li>Monitoring your symptoms closely</li>
                                <li>Having a plan in case symptoms worsen</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)

            st.markdown("<h3>Nutrition & Hydration Tips</h3>", unsafe_allow_html=True)
            st.markdown("""
                <div class='card'>
                    <h4>Foods that may help combat air pollution effects:</h4>
                    <ul>
                        <li><strong>Antioxidant-rich foods:</strong> Berries, dark leafy greens, and colorful vegetables</li>
                        <li><strong>Omega-3 sources:</strong> Fatty fish, walnuts, and flaxseeds to reduce inflammation</li>
                        <li><strong>Vitamin C:</strong> Citrus fruits, bell peppers, and broccoli to boost immune function</li>
                        <li><strong>Vitamin E:</strong> Nuts, seeds, and olive oil to protect cells from oxidative damage</li>
                    </ul>
                    <h4>Hydration guidelines:</h4>
                    <ul>
                        <li>Drink 16-20 oz of water 2 hours before outdoor activity</li>
                        <li>Consume 7-10 oz of fluid every 10-20 minutes during activity</li>
                        <li>Consider electrolyte drinks for activities longer than 60 minutes</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
            if not df.empty and 'Air Quality Index (AQI)' in df.columns:
                current_aqi = df['Air Quality Index (AQI)'].iloc[167]
                st.markdown("<h3>General Precautions</h3>", unsafe_allow_html=True)
                
                if current_aqi >= 80 and current_aqi <= 100:
                    bg_color = "#009E3A"
                    st.markdown(f"""
                        <div class='warning' style="background: linear-gradient(to right, {bg_color}, {bg_color}CC); color: white; border-left: 0.5rem solid #007D2E;">
                            <h4>Excellent Air Quality (AQI: {current_aqi:.1f})</h4>
                            <p>Current air quality is excellent. Enjoy your outdoor activities!</p>
                            <ul>
                                <li>Perfect conditions for your planned activity</li>
                                <li>Stay hydrated</li>
                                <li>Enjoy the fresh air</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                elif current_aqi >= 60 and current_aqi < 80:
                    bg_color = "#84CF33"
                    st.markdown(f"""
                        <div class='warning' style="background: linear-gradient(to right, {bg_color}, {bg_color}CC); color: white; border-left: 0.5rem solid #6BAA29;">
                            <h4>Good Air Quality (AQI: {current_aqi:.1f})</h4>
                            <p>Current air quality is good. Proceed with your outdoor plans.</p>
                            <ul>
                                <li>Good conditions for outdoor activities</li>
                                <li>Stay hydrated</li>
                                <li>Monitor how you feel during activity</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                elif current_aqi >= 40 and current_aqi < 60:
                    bg_color = "#FFFF00"
                    st.markdown(f"""
                        <div class='warning' style="background: linear-gradient(to right, {bg_color}, {bg_color}CC); color: black; border-left: 0.5rem solid #CCCC00;">
                            <h4>Moderate Air Quality (AQI: {current_aqi:.1f})</h4>
                            <p>Current air quality is moderate. Some precautions recommended.</p>
                            <ul>
                                <li>Sensitive individuals should take breaks as needed</li>
                                <li>Stay well hydrated</li>
                                <li>Monitor any symptoms if you have pre-existing conditions</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                elif current_aqi >= 20 and current_aqi < 40:
                    bg_color = "#FF8C00"
                    st.markdown(f"""
                        <div class='warning' style="background: linear-gradient(to right, {bg_color}, {bg_color}CC); color: white; border-left: 0.5rem solid #CC7000;">
                            <h4>Low Air Quality (AQI: {current_aqi:.1f})</h4>
                            <p>Current air quality is low. Take precautions.</p>
                            <ul>
                                <li>Consider reducing intensity of outdoor activities</li>
                                <li>Take frequent breaks</li>
                                <li>Consider a mask if you have respiratory issues</li>
                                <li>Stay well hydrated</li>
                                <li>Monitor symptoms closely</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                elif current_aqi >= 0 and current_aqi < 20:
                    bg_color = "#FF0000"
                    st.markdown(f"""
                        <div class='warning' style="background: linear-gradient(to right, {bg_color}, {bg_color}CC); color: white; border-left: 0.5rem solid #CC0000;">
                            <h4>Poor Air Quality (AQI: {current_aqi:.1f})</h4>
                            <p>Current air quality is poor. Significant precautions advised.</p>
                            <ul>
                                <li>Consider wearing an N95 mask outdoors</li>
                                <li>Limit outdoor exposure when possible</li>
                                <li>Keep windows closed</li>
                                <li>Use air purifiers indoors</li>
                                <li>Consider rescheduling strenuous outdoor activities</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                elif current_aqi == 0:
                    bg_color = "#800000"
                    st.markdown(f"""
                        <div class='warning' style="background: linear-gradient(to right, {bg_color}, {bg_color}CC); color: white; border-left: 0.5rem solid #660000;">
                            <h4>Very Poor Air Quality (AQI: {current_aqi:.1f})</h4>
                            <p>Current air quality is very poor. Strong precautions advised.</p>
                            <ul>
                                <li>Wear an N95 mask if outdoor activity is necessary</li>
                                <li>Stay indoors as much as possible</li>
                                <li>Keep all windows closed</li>
                                <li>Use air purifiers</li>
                                <li>Strongly consider postponing outdoor activities</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    bg_color = "#3498db"
                    st.markdown(f"""
                        <div class='card' style="background: linear-gradient(to right, {bg_color}, {bg_color}CC); color: white;">
                            <h4>Air Quality (AQI: {current_aqi:.1f})</h4>
                            <p>Current air quality information. Enjoy your outdoor activities while staying hydrated and being mindful of your body's signals.</p>
                        </div>
                    """, unsafe_allow_html=True)
        
        with tab3:
            st.header("Mask Recommendations")
            
            if st.session_state.mask_recommendations is None:
                mask_btn = st.button("Get Mask Recommendations")
                
                if mask_btn:
                    with st.spinner("Our AI agent is finding the best masks for air quality protection..."):
                        try:
                            current_aqi = df['Air Quality Index (AQI)'].iloc[167] if not df.empty and 'Air Quality Index (AQI)' in df.columns else "unknown"
                            
                            mask_results = get_mask_recommendations()
                            
                            st.session_state.mask_recommendations = {
                                'results': mask_results,
                                'aqi': current_aqi
                            }
                            
                            st.success("Mask recommendations generated successfully!")
                        except Exception as e:
                            st.error(f"Error generating mask recommendations: {str(e)}")
            
            if st.session_state.mask_recommendations:
                mask_data = st.session_state.mask_recommendations
                current_aqi = mask_data['aqi']
                mask_results = mask_data['results']
                
                if current_aqi != "unknown":
                    if current_aqi < 60: 
                        st.markdown(f"""
                            <div class='warning' style="background-color: #dc3545; border-left: 0.5rem solid #dc3545;">
                                <h4>Mask Recommended</h4>
                                <p>With the current AQI of {current_aqi:.1f}, wearing a mask during outdoor activities is recommended, 
                                especially if you have respiratory conditions or plan extended outdoor exposure.</p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class='warning' style="background-color: #198754; border-left: 0.5rem solid #198754;">
                                <h4>Masks Optional</h4>
                                <p>With the current AQI of {current_aqi:.1f}, masks are generally optional for most people. 
                                However, if you have respiratory conditions or are particularly sensitive to air pollution, 
                                consider using a mask during extended outdoor activities.</p>
                            </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("<h3>Expert Mask Recommendations</h3>", unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div class='mask-card'>
                        {str(mask_results).replace(chr(10), '<br>')}
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<h3>Mask Types for Air Quality Protection</h3>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("""
                        <div class='mask-card'>
                            <h4>N95 Respirators</h4>
                            <p><strong>Filtration:</strong> 95% of airborne particles</p>
                            <p><strong>Best for:</strong> High pollution days, wildfire smoke</p>
                            <p><strong>Features:</strong> Tight seal, NIOSH certified</p>
                            <p><strong>Limitations:</strong> Can be uncomfortable for long use, requires proper fitting</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                        <div class='mask-card'>
                            <h4>KN95 Masks</h4>
                            <p><strong>Filtration:</strong> 95% of airborne particles</p>
                            <p><strong>Best for:</strong> Daily use in polluted areas</p>
                            <p><strong>Features:</strong> Generally more comfortable than N95</p>
                            <p><strong>Limitations:</strong> Quality can vary between manufacturers</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                        <div class='mask-card'>
                            <h4>Surgical Masks</h4>
                            <p><strong>Filtration:</strong> Moderate protection</p>
                            <p><strong>Best for:</strong> Light pollution, casual use</p>
                            <p><strong>Features:</strong> Lightweight, comfortable</p>
                            <p><strong>Limitations:</strong> Less effective for fine particles</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<h3>Mask Care & Usage Tips</h3>", unsafe_allow_html=True)
                st.markdown("""
                    <div class='mask-card'>
                        <ul>
                            <li><strong>Proper Fit:</strong> Ensure the mask creates a seal around your nose and mouth</li>
                            <li><strong>Replacement:</strong> Replace disposable masks after each use or when they become damp/dirty</li>
                            <li><strong>Reusable Masks:</strong> Wash cloth masks after each use</li>
                            <li><strong>Storage:</strong> Store clean masks in a breathable container when not in use</li>
                            <li><strong>Hand Hygiene:</strong> Wash hands before putting on and after removing your mask</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            else:
                if not mask_btn:
                    st.info("Click the 'Get Mask Recommendations' button to receive personalized mask recommendations from our AI agent.")
        with tab4:
            st.header("Historical AQI Data")
            
            if not df.empty:
                display_df = df.copy()
                if 'ds' in display_df.columns:
                    display_df.rename(columns={'ds': 'Date'}, inplace=True)
                st.subheader("AQI Trend")
                if 'Air Quality Index (AQI)' in display_df.columns and not display_df.empty:
                    chart_data = display_df[['date', 'Air Quality Index (AQI)']]
                    chart_data = chart_data.dropna()
                    if not chart_data.empty:
                        chart_data['date'] = pd.to_datetime(chart_data['date'])
                        chart_data = chart_data.sort_values('date')
                        chart_data['date'] = chart_data['date'].dt.strftime('%Y-%m-%d %H:%M')
                        st.line_chart(chart_data.set_index('date')['Air Quality Index (AQI)'])
                    
                    st.subheader("AQI Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        avg_aqi = display_df['Air Quality Index (AQI)'].mean()
                        avg_color = color_aqi(avg_aqi)
                        st.markdown(
                            f"""
                            <div style="padding: 1rem; border-radius: 0.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);">
                                <div style="font-size: 0.875rem; color: rgb(100, 116, 139);">Average AQI</div>
                                <div style="font-size: 1.875rem; {avg_color};">{avg_aqi:.1f}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with col2:
                        max_aqi = display_df['Air Quality Index (AQI)'].max()
                        max_color = color_aqi(max_aqi)
                        st.markdown(
                            f"""
                            <div style="padding: 1rem; border-radius: 0.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);">
                                <div style="font-size: 0.875rem; color: rgb(100, 116, 139);">Maximum AQI</div>
                                <div style="font-size: 1.875rem; {max_color};">{max_aqi:.1f}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with col3:
                        min_aqi = display_df['Air Quality Index (AQI)'].min()
                        min_color = color_aqi(min_aqi)
                        st.markdown(
                            f"""
                            <div style="padding: 1rem; border-radius: 0.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);">
                                <div style="font-size: 0.875rem; color: rgb(100, 116, 139);">Minimum AQI</div>
                                <div style="font-size: 1.875rem; {min_color};">{min_aqi:.1f}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                if 'y' in display_df.columns:
                    display_df.rename(columns={'y': 'Air Quality Index (AQI)'}, inplace=True)
                
                st.dataframe(
                    display_df.style.applymap(color_aqi, subset=['Air Quality Index (AQI)']),
                    use_container_width=True,
                    hide_index=True
                )
                
                csv = display_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Data as CSV",
                    data=csv,
                    file_name=f"aqi_data_{city}_{country}.csv",
                    mime="text/csv",
                )
            else:
                st.info("No historical data available.")
        
if __name__ == "__main__":
    main()
