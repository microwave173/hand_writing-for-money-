from flask import Flask, request, send_from_directory, abort
import os

app = Flask(__name__)


@app.route('/getfile', methods=['GET'])
def get_file():
    try:
        # 获取请求的文件名参数
        filename = request.args.get('filename', '')

        # 获取当前脚本所在目录
        current_directory = os.path.dirname(os.path.abspath(__file__))

        # 完整的文件路径
        file_path = os.path.join(current_directory, filename)

        # 检查文件是否存在
        if not os.path.isfile(file_path):
            abort(404)  # 如果文件不存在，则返回404未找到

        # 返回请求的文件内容
        return send_from_directory(current_directory, filename)

    except Exception as e:
        return str(e), 500


@app.route('/hello', methods=['GET'])
def hello():
    return 'hello world'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
