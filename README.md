# AQI_AI_Agent
repo contains code on building AQI AI Agent using FIRECRAWL API and OPEN AI API

conda create -n aqiagent python=3.11 -y

Firecrawl API Key here : https://www.firecrawl.dev/app/api-keys


The AQI Analysis Agent is a powerful air quality monitoring and health recommendation tool powered by Firecrawl and Agno's AI Agent framework. This app helps users make informed decisions about outdoor activities by analyzing real-time air quality data and providing personalized health recommendations.

Features : 

Multi-Agent System: 
AQI Analyzer: Fetches and processes real-time air quality data
Health Recommendation Agent: Generates personalized health advice


Air Quality Metrics:
Overall Air Quality Index (AQI)
Particulate Matter (PM2.5 and PM10)
Carbon Monoxide (CO) levels
Temperature
Humidity
Wind Speed


Comprehensive Analysis:
Real-time data visualization
Health impact assessment
Activity safety recommendations
Best time suggestions for outdoor activities
Weather condition correlations


Interactive Features:
Location-based analysis
Medical condition considerations
Activity-specific recommendations
Downloadable reports
Example queries for quick testing


-----------------------------------------------------------------------------

Usage: 

Enter your API keys in the API Configuration section
Input location details:
City name
State (optional for Union Territories/US cities)
Country
Provide personal information:
Medical conditions (optional)
Planned outdoor activity
Click "Analyze & Get Recommendations" to receive:
Current air quality data
Health impact analysis
Activity safety recommendations
Try the example queries for quick testing

Note :
The air quality data is fetched using Firecrawl's web scraping capabilities. Due to caching and rate limiting, the data might not always match real-time values on the website. For the most accurate real-time data, consider checking the source website directly.