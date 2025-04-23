import asyncio
import threading
import csv
from datetime import datetime

import numpy as np

from flask import Flask, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS

from service.device1_service import receive_heart_rate, gen_data, get_data_metrics, get_time_interval_per_point, \
    get_heart_rate_variability_scatter_plot

toggle = False
dataList = []
csv_file = 'data.csv'
global writer, file


def get_time():
    # 获取当前时间
    current_time = datetime.now()
    # 将时间转换为指定格式的字符串
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    return formatted_time


def callback_(data: dict):
    socketio.emit('device1_data_upload', data)
    if toggle:
        data['time'] = datetime.now()
        dataList.append(data)
        row = [get_time()]
        row.extend(data.values())
        writer.writerow([str(item) for item in row])


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)


def handle_connect():
    # 创建一个事件循环
    loop = asyncio.new_event_loop()
    # 在事件循环中运行异步函数
    asyncio.set_event_loop(loop)
    task = loop.create_task(receive_heart_rate(callback_))
    # 执行事件循环，直到异步函数执行完成
    loop.run_until_complete(task)


@app.route('/start')
def start():
    global toggle, dataList, writer, file
    toggle = True
    dataList = []
    data = {
        'message': 'success',
        'code': 0,
        'dataList': dataList
    }
    file = open(csv_file, mode='w', newline='', encoding='utf-8')
    writer = csv.writer(file)
    writer.writerow(['时间戳', '血压舒张压', '血压伸缩压', '心率', '微循环', '血氧饱和度', '脉搏波', 'x轴', 'y轴', 'z轴'])
    return jsonify(data)


@app.route('/record')
def record():
    if len(dataList) > 0:

        pulse_wave_strength = [item['pulse_wave_strength'] for item in dataList]
        pulse_wave_strength = np.ravel(pulse_wave_strength)

        sampling_interval = get_time_interval_per_point(len(pulse_wave_strength), dataList[0]['time'],
                                                        dataList[-1]['time'])
        print(get_data_metrics(pulse_wave_strength, dataList[0]['time'], dataList[-1]['time']).items())

        # gen_data(pulse_wave_strength.tolist())
        pulse_wave_strength = np.array(pulse_wave_strength)
        # 快速傅立叶变换（FFT）
        fft_result = np.fft.fft(pulse_wave_strength)
        freqs = np.fft.fftfreq(len(pulse_wave_strength), sampling_interval)

        heart_rate_data = [item['heart_rate'] for item in dataList]
        heart_rate_data = np.array(heart_rate_data)
        heart_rate_data = np.sort(heart_rate_data)
        heart_rate_variability_scatter_plot = get_heart_rate_variability_scatter_plot(pulse_wave_strength, dataList[0]['time'], dataList[-1]['time'])

        data = {
            'message': 'success',
            'code': 0,
            'fft_result': np.abs(fft_result).tolist(),
            'freqs': freqs.tolist(),
            'heart_rate_variability_scatter_plot': heart_rate_variability_scatter_plot,
            'metrics': get_data_metrics(pulse_wave_strength, dataList[0]['time'], dataList[-1]['time'])
        }
    else:
        data = {
            'message': 'success',
            'code': 0,
            'fft_result': [],
            'freqs': [],
            'lorenz_heart_rate_data': [],
            'metrics': {}
        }
    return jsonify(data)


@app.route('/stop')
def stop():
    global toggle, writer, file
    toggle = False
    data = {
        'message': 'success',
        'code': 0,
        'dataList': dataList
    }
    if len(dataList) > 0:
        pulse_wave_strength = [item['pulse_wave_strength'] for item in dataList]
        pulse_wave_strength = np.ravel(pulse_wave_strength)
        items = get_data_metrics(pulse_wave_strength, dataList[0]['time'], dataList[-1]['time']).items()
        writer.writerows([[item[0].upper(), item[1]] for item in items])
        file.close()
    return jsonify(data)


if __name__ == '__main__':
    thread = threading.Thread(target=handle_connect)
    thread.start()
    # 启动 WebSocket 服务
    socketio.run(app, allow_unsafe_werkzeug=True)
