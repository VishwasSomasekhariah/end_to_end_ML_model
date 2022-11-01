""" This file creates a server for the client to interact with."""

from concurrent import futures
import grpc
import model_pb2
import model_pb2_grpc
import socket
import tensorflow as tf
from tensorflow import keras


def normalize_img(image):
    return tf.cast(image, tf.float32) / 255.0


def load_image(image_bytes):
    image_tensor = tf.io.decode_raw(image_bytes, tf.uint8)
    image_tensor = tf.reshape(image_tensor, [1, 28, 28])
    image_tensor = normalize_img(image_tensor)
    return image_tensor


def predict(image_bytes):
    model = keras.models.load_model("mnist_model")
    image = load_image(image_bytes)
    result = model.predict(image)[0]
    print("All Prediction results", result)
    return result.argmax(), result[result.argmax()] * 100


class Predictor(model_pb2_grpc.PredictorServicer):
    def predictImage(self, request, context):
        guess, confidence = predict(image_bytes=request.processedImage)
        print(f"Server Guess {guess}, Confidence {confidence}")
        return model_pb2.predictionResponse(
            guess=int(guess), confidence=float(confidence)
        )


def serve():
    print("Server Running...")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_pb2_grpc.add_PredictorServicer_to_server(Predictor(), server)
    server.add_insecure_port("[::]:9999")
    print("Server located at: ", end="")
    print(socket.gethostbyname(socket.gethostname()))

    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
