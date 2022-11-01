from flask import Flask
from PIL import Image

import grpc
import model_pb2
import model_pb2_grpc

app = Flask("end_to_end_ML_model_client")


@app.route("/")
def run_client():
    try:
        with grpc.insecure_channel("172.17.0.2:9999") as channel:
            stub = model_pb2_grpc.PredictorStub(channel)
            # Test image is in color(RGBA) mode need to first convert it to grayscale("L") mode and then to bytes.
            bytes_string = Image.open("test_image11.png").convert("L").tobytes("raw", "L")

            response = stub.predictImage(
                model_pb2.modelRequest(processedImage=bytes_string)
            )

            print("Client Recieved the response successfully:" + str(response))
            return f"""<h1>Client Received:</h1>
            <h2>Guess: '{str(response.guess)}'</h2>
            <h2>Confidence: {response.confidence:.3f}%</h2>"""
    except Exception as _e:
        return "<h1>ERROR</h1>" + str(_e)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
