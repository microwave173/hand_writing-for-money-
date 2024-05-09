from flask import Flask, request, jsonify
import json
from datetime import datetime
import os

app = Flask(__name__)


@app.route('/postjson', methods=['POST'])
def post_json():
    try:
        # 确保请求中有JSON数据
        data = request.get_json()
        if data is None:
            return jsonify({"error": "No JSON received"}), 400

        # 获取当前时间戳
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

        # 创建存储JSON文件的目录（如果不存在）
        os.makedirs('json_data', exist_ok=True)

        # 将JSON数据保存到文件
        with open(f'json_data/{timestamp}.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)

        return jsonify({"message": "200"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
