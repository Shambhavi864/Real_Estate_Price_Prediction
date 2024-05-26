import util
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load artifacts at the start
util.load_saved_artifacts()

@app.route('/get_location_names')
def get_location_names():
    locations = util.get_location_names()
    print(f"Locations fetched: {locations}")  # Debugging statement
    response = jsonify({
        'locations': locations
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    response = jsonify({
        'estimated_price': util.get_estimated_price(location, total_sqft, bhk, bath)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == "__main__":
    print("Python Bangalore price prediction model...")
    app.run()
