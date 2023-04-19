# -----Imports-----
import json

from flask import Flask, request, jsonify
from flask_cors import CORS

from UseModel import generate_prediction
from PreProcess import process_hrv_data

# -----Constants-----
json_path = "./Database/Data.json"

# -----App initalization-----
app = Flask(__name__)
CORS(app)

# -----API endpoints-----
# Read data
@app.route('/data/read', methods=['GET'])
def read_data():
    # Open the json file
    json_file = open(json_path)

    # Convert to dict
    data = json.load(json_file)

    return (data), 200

# Update data
@app.route('/data/update', methods=['GET'])
def update_data():
    # Get data
    req_from = request.args.get("req_from")
    status = request.args.get("status")
    reading = request.args.get("reading")

    # Variables
    prediction = ""

    # Check params are empty
    if None not in (req_from, status, reading):
        # Convert reading to list
        reading_list = json.loads(reading)

        # Check param values valid
        if(req_from == "device" and status == "2" and len(reading_list) > 0):
            # Preprocess heart rate variability data
            hr_mean_raw, hr_std_raw = process_hrv_data(reading_list)

            # Get prediction
            prediction = generate_prediction(float(hr_mean_raw), float(hr_std_raw))
        else:
            return jsonify("Invalid parameter values for device!"), 400
    elif req_from is None and reading is None and status is not None:
        if status == "0" or status == "1":
            prediction = "4"
        else:
            return jsonify("Invalid status parameter for moibile app!"), 400
    else:
        return jsonify("Provide valid request parameters!"), 400

    # Read the json file
    with open(json_path, "r") as json_file:
        data = json.load(json_file)

    # Update content
    data["status"] = status
    data["prediction"] = prediction

    # Save the json file
    with open(json_path, "w") as json_file:
        json.dump(data, json_file)

    return jsonify("success"), 200

# Execute the app
if __name__ == "__main__":
    app.run(host="0.0.0.0")