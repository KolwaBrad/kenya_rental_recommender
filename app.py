from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from markupsafe import Markup
from src.data_preprocessing import prepare_data
from src.model import train_model, find_optimal_clusters
from src.recommender import recommend_neighborhood
from src.utils import prepare_results, format_gemini_response, clean_property_description
import requests
import feedparser
from urllib.parse import quote
import time
import googlemaps
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv
from flask_session import Session

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')

# Configure server-side session
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

gmaps = googlemaps.Client(key=os.getenv('GOOGLE_MAPS_API_KEY'))
genai.configure(api_key=os.getenv('GOOGLE_GEMINI_API_KEY'))
model_gemini = genai.GenerativeModel('gemini-pro')

df, X, scaler, imputer = prepare_data('data/kenya_rentals.csv')
optimal_clusters = find_optimal_clusters(X)
model = train_model(X, n_clusters=optimal_clusters)

def get_location_coordinates(address, neighborhood):
    try:
        full_address = f"{address}, {neighborhood}, Kenya"
        geocode_result = gmaps.geocode(full_address)
        
        if geocode_result:
            location = geocode_result[0]['geometry']['location']
            return location['lat'], location['lng']
        return None
    except Exception as e:
        print(f"Error getting coordinates: {e}")
        return None

def search_nearby_places(location, place_type, radius=2000, max_results=2):
    try:
        places_result = gmaps.places_nearby(
            location=location,
            radius=radius,
            type=place_type
        )
        
        places = []
        if places_result.get('results'):
            for place in places_result['results'][:max_results]:
                place_details = {
                    'name': place.get('name'),
                    'rating': place.get('rating', 'N/A'),
                }
                places.append(place_details)
        
        return places
    except Exception as e:
        print(f"Error searching nearby places: {e}")
        return []

def get_amenities_info(neighborhood, address):
    amenity_types = {
        'Schools': 'school',
        'Hospitals': 'hospital',
        'Restaurants': 'restaurant',
        'Supermarkets': 'supermarket',
        'Banks': 'bank'
    }
    
    coordinates = get_location_coordinates(address, neighborhood)
    if not coordinates:
        return "Unable to find location coordinates."
    
    amenities_dict = {}
    for category_name, amenity_type in amenity_types.items():
        places = search_nearby_places(coordinates, amenity_type)
        amenities_dict[category_name] = places
    
    return amenities_dict

def get_neighborhood_news(neighborhood, num_news=2):
    try:
        encoded_query = quote(f"{neighborhood} Kenya news")
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        feed = feedparser.parse(response.content)
        
        news_items = []
        for entry in feed.entries[:num_news]:
            news_items.append({
                'title': entry.title,
                'link': entry.link,
                'date': ' '.join(entry.published.split()[1:5])
            })
        
        return news_items
        
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

def generate_ai_recommendations(properties):
    """
    Generate AI recommendations using Google's Gemini API based on property data
    """
    try:
        # Prepare the prompt
        prompt = "Based on the following neighborhoods in Kenya, recommend the top 2 best options. Consider factors like amenities, recent news, and property features. Provide detailed reasoning for each recommendation. Your response should be well-formatted with clear paragraphs and proper spacing.\n\n"
        
        for idx, prop in enumerate(properties, 1):
            prompt += f"\nProperty {idx} - {prop['Neighborhood']}:\n"
            prompt += f"Price: KSh {prop['Price']}\n"
            prompt += f"Specs: {prop['Bedrooms']} bedrooms, {prop['Bathrooms']} bathrooms, {prop['sq_mtrs']} sq meters\n"
            
            # Add amenities information
            prompt += "Nearby Amenities:\n"
            for category, places in prop['amenities'].items():
                if places:
                    prompt += f"- {category}: {', '.join(f'{place['name']} (Rating: {place['rating']})' for place in places)}\n"
            
            # Add recent news
            if prop['news']:
                prompt += "Recent News:\n"
                for news in prop['news']:
                    prompt += f"- {news['title']}\n"
            
            prompt += "\n"

        # Get recommendation from Gemini
        response = model_gemini.generate_content(prompt)
        
        # Format the response
        formatted_recommendations = format_gemini_response(response.text)
        
        # Filter and format properties
        recommended_neighborhoods = []
        for prop in properties:
            if prop['Neighborhood'] in response.text:
                # Create a description for the property
                prop_description = (
                    f"**Location Details**: {prop['Neighborhood']}\n\n"
                    f"**Property Features**: {prop['Bedrooms']} bedrooms, "
                    f"{prop['Bathrooms']} bathrooms, {prop['sq_mtrs']} square meters\n\n"
                )
                prop['description'] = clean_property_description(prop_description)
                recommended_neighborhoods.append(prop)

        return {
            'success': True,
            'recommendations': formatted_recommendations,
            'properties': recommended_neighborhoods[:2]
        }
    except Exception as e:
        print(f"Error generating AI recommendations: {e}")
        return {
            'success': False,
            'error': str(e)
        }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = {
            'budget': float(request.form['budget']),
            'bedrooms': int(request.form['bedrooms']),
            'bathrooms': int(request.form['bathrooms']),
            'sq_mtrs': float(request.form['sq_mtrs'])
        }
        
        recommended_neighborhoods, recommended_properties = recommend_neighborhood(
            user_input, model, df, X, scaler, imputer
        )
        
        formatted_properties = prepare_results(recommended_properties)
        
        for property in formatted_properties:
            property['amenities'] = get_amenities_info(
                property['Neighborhood'], 
                property.get('Address', f"{property['Neighborhood']}, Nairobi")
            )
            property['news'] = get_neighborhood_news(property['Neighborhood'])
            time.sleep(1)
        
        session['properties'] = formatted_properties
        
        return render_template(
            'results.html',
            neighborhoods=recommended_neighborhoods,
            properties=formatted_properties
        )
    
    return render_template('index.html')

@app.route('/ai-recommendations')
def ai_recommendations():
    try:
        properties = session.get('properties')
        
        if not properties:
            return redirect(url_for('index'))
        
        ai_results = generate_ai_recommendations(properties)
        
        if not ai_results['success']:
            return render_template(
                'ai_recommendations.html',
                error=ai_results.get('error'),
                recommendations=None,
                recommended_properties=[]
            )
        
        return render_template(
            'ai_recommendations.html',
            recommendations=ai_results['recommendations'],
            recommended_properties=ai_results['properties'],
            error=None
        )
        
    except Exception as e:
        print(f"Error in ai_recommendations route: {e}")
        return render_template(
            'ai_recommendations.html',
            error="An unexpected error occurred. Please try again.",
            recommendations=None,
            recommended_properties=[]
        )

if __name__ == '__main__':
    app.run(debug=True)