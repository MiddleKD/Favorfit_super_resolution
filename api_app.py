from flask import Flask, request
from lambda_function import lambda_handler
from utils import load_bs64_from_url

app = Flask(__name__)

@app.route('/')
def call_main():
    return 'Hello, This is super resolution'

@app.route('/sample/', methods=["GET"])
def call_handler():
    image_bs64 = load_bs64_from_url("https://img.danawa.com/prod_img/500000/107/815/img/3815107_1.jpg?shrink=330:*&_v=20230314162424")
    res = lambda_handler({"body":{"image_b64":"data:application/octet-stream;base64,"+image_bs64, "avail_gpu":False}}, None)
    return res

@app.route('/post/bs64', methods=["POST"])
def post_bs64_handler():
    data = request.get_json()
    res = lambda_handler(data, None)
    return res

@app.route('/post/url', methods=["POST"])
def post_url_handler():
    data = request.get_json()
    url = data.get("url")
    avail_gpu = data.get("gpu")
    
    image_bs64 = load_bs64_from_url(url)
    res = lambda_handler({"image_b64":image_bs64, "avail_gpu":avail_gpu}, None)
    return res

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False, port=8000)
