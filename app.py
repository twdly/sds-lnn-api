from flask import Flask
from flask import Response
from flask import request

import state

app = Flask(__name__)


@app.route("/get", methods=['GET'])
def get_json():
    return state.get_state()


# This method will need to be secured somehow in the future to prevent unauthorised people from sending in weather data
@app.route("/set", methods=['POST'])
def set_state():
    form = request.form
    month = form.get("month")
    day = form.get("day")
    hour = form.get("hour")
    lat = form.get("lat")
    lon = form.get("lon")
    wind = form.get("wind")
    pres = form.get("pres")
    gust = form.get("gust")
    eye = form.get("eye")
    speed = form.get("speed")
    state.update_state(month, day, hour, lat, lon, wind, pres, gust, eye, speed)
    return Response(status=200)


@app.route("/get-history", methods=['GET'])
def get_history():
    args = request.args
    return state.get_history(int(args.get("count")))


if __name__ == "__main__":
    Flask.run(app)
