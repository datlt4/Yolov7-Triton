from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS, cross_origin
from request_yolov7_trt import *
import numpy as np
import cv2

app = Flask(__name__)
CORS(app, support_credentials=True)
client_yolov7 = TritonClientYolov7()


@app.route("/", methods=["GET"])
@cross_origin(supports_credentials=True)
def root():
    return "Yolov7 Triton server"


@app.route("/inference", methods=["POST"])
@cross_origin(supports_credentials=True)
def inference():
    filestr = request.files["file"]
    file_bytes = np.fromstring(filestr.read(), np.uint8)
    # convert numpy array to image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    result = client_yolov7.triton_infer(image)
    bbox = []
    for r in result:
        bbox.append(r.to_json(client_yolov7.class_id))
    return {"bounding_box": bbox}


if __name__=="__main__":
    app.run(host="0.0.0.0", port=8003, ssl_context='adhoc')


