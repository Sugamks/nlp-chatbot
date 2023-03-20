
from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request, jsonify
from chat import get_response
from kb import response
app = Flask(__name__)
run_with_ngrok(app) 
@app.route("/", methods=["GET"])
def index_get():
    return render_template("base.html")


@app.route("/predict",methods=["POST"])
def predict():
    text = request.get_json().get("message")
    qa_response = qaprocessing(text)
    if response != "I do not understand...":
      message = {"answer": qa_response}
      return jsonify (message)
    elif response == "I do not understand...":
	kb_response = kbprocessing(text)
      message = {"answer": kb_response}
      return jsonify (message) 
    else:
      return "Sorry I cant answer"

if __name__ == "__main__":
    app.run()