from flask import Flask
from flask import Response
from flask import request

import lnn

app = Flask(__name__)


@app.route("/get", methods=['GET'])
def get_json():
    return lnn.get_state()


# This method will need to be secured somehow in the future to prevent unauthorised people from sending in weather data
@app.route("/set", methods=['POST'])
def set_state():
    form = request.form
    temperature = form.get("temperature")
    vorticity = form.get("vorticity")
    lnn.update_state(vorticity, temperature)
    return Response(status=200)


@app.route("/get-history", methods=['GET'])
def get_history():
    args = request.args
    return lnn.get_history(int(args.get("count")))


if __name__ == "__main__":
    Flask.run(app)
