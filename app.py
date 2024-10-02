from flask import Flask
from flask import request

import lnn

app = Flask(__name__)


@app.route("/", methods=['GET'])
def get_json():
    state = lnn.get_state()
    prediction = lnn.get_prediction()
    return f"Prediction: {prediction}, vorticity: {state[0]}, temperature: {state[1]}"


# This method will need to be secured somehow in the future to prevent unauthorised people from sending in weather data
@app.route("/set", methods=['POST'])
def set_state():
    args = request.args
    temperature = args.get("temp")
    vorticity = args.get("vort")
    lnn.update_state(vorticity, temperature)
    return 'success'


if __name__ == "__main__":
    Flask.run(app)