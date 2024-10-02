from flask import Flask
from flask import Response
from flask import request

import lnn

app = Flask(__name__)


@app.route("/", methods=['GET'])
def get_json():
    return lnn.get_state()


# This method will need to be secured somehow in the future to prevent unauthorised people from sending in weather data
@app.route("/set", methods=['POST'])
def set_state():
    args = request.form
    temperature = args.get("temperature")
    vorticity = args.get("vorticity")
    lnn.update_state(vorticity, temperature)
    return Response(status=200)


if __name__ == "__main__":
    Flask.run(app)
