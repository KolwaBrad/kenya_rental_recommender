# Kenya Residential Relocation Recommender

A machine learning-powered system that recommends optimal neighborhoods in Kenya for residential relocation, enhanced with Google Maps data and Gemini AI insights.

## Features

- Machine learning-based neighborhood recommendations
- Integration with Google Maps API for detailed location data
- Advanced filtering using Google's Gemini AI
- Interactive web interface built with Flask
- Data-driven insights about neighborhoods
- Real-time location data processing

## Tech Stack

- Python 3.8+
- Flask web framework
- Scikit-learn for machine learning
- Google Maps API
- Google Gemini AI
- Bootstrap for frontend styling

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/kenya-relocation-recommender.git
cd kenya-relocation-recommender
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # For Unix/macOS
venv\Scripts\activate     # For Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory with:
```
GOOGLE_MAPS_API_KEY=your_google_maps_api_key
GOOGLE_GEMINI_API_KEY=your_gemini_api_key
```

## Project Structure

```
kenya-relocation-recommender/
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── recommender.py
│   └── utils.py
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   └── js/
├── app.py
├── requirements.txt
└── README.md
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Enter your preferences and receive personalized neighborhood recommendations

## How It Works

1. **Data Collection**: The system collects neighborhood data through the Google Maps API, including:
   - Points of interest
   - Transportation options
   - Safety metrics
   - Amenities

2. **Machine Learning Processing**: 
   - Processes user preferences
   - Applies clustering algorithms
   - Generates initial recommendations

3. **AI Enhancement**:
   - Top 5 neighborhoods are analyzed by Google's Gemini AI
   - Additional context and insights are generated
   - Final 2 recommendations are selected based on AI analysis

4. **Result Presentation**:
   - Interactive map display
   - Detailed neighborhood information
   - AI-generated insights and comparisons

## API Keys Required

- Google Maps API Key
- Google Gemini AI API Key

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Website - Soon

Project Link: [https://github.com/yourusername/kenya-relocation-recommender](https://github.com/yourusername/kenya-relocation-recommender)