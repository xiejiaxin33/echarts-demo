import asyncio

import bleak
import numpy as np
from bleak import BleakGATTCharacteristic
from scipy.signal import welch


class BLEDevice1:

    def __load_write_and_notify(self):
        for service in self.client.services:
            if service.description != 'Nordic UART Service': continue
            characteristics = service.characteristics
            # 遍历服务中的所有特征值
            for characteristic in characteristics:
                if 'write' in characteristic.properties:
                    self.__write_uuid = characteristic.uuid
                if 'notify' in characteristic.properties:
                    self.__notify_uuid = characteristic.uuid
        assert self.__write_uuid is not None
        assert self.__notify_uuid is not None

    def __init__(self, device_uuid):
        self.client = bleak.BleakClient(device_uuid)
        self.__write_uuid = None
        self.__notify_uuid = None

    async def connect(self):
        await self.client.connect()
        self.__load_write_and_notify()

    async def write(self, data, **kwargs):
        await self.client.write_gatt_char(self.__write_uuid, data=data, **kwargs)

    async def notify(self, callback_, **kwargs):
        await self.client.start_notify(self.__notify_uuid, callback_, **kwargs)


class ProtocolDevice1:

    def __init__(self, data: bytearray):
        assert len(data) == 182
        self.blood_pressure_systolic = int.from_bytes(data[2:3], byteorder='little', signed=False)
        self.blood_pressure_diastolic = int.from_bytes(data[4:5], byteorder='little', signed=False)
        self.microcirculation = int.from_bytes(data[6:7], byteorder='little', signed=False)
        self.heart_rate = int.from_bytes(data[8:9], byteorder='little', signed=False)
        self.oxygen_saturation = int.from_bytes(data[10:11], byteorder='little', signed=False)
        self.pulse_wave_strength = [int.from_bytes(data[i:i+1], byteorder='little', signed=True) for i in range(12, 75)]

        self.axisX = [int.from_bytes(data[(80+i*6):(81+i*6)], byteorder='little', signed=False) for i in range(0, 16)]
        self.axisY = [int.from_bytes(data[(82+i*6):(83+i*6)], byteorder='little', signed=False) for i in range(0, 16)]
        self.axisZ = [int.from_bytes(data[(84+i*6):(85+i*6)], byteorder='little', signed=False) for i in range(0, 16)]


async def receive_heart_rate(callback_):
    def callback_new_(sender: BleakGATTCharacteristic, data: bytearray):
        data = ProtocolDevice1(data)
        callback_(data.__dict__)
    device = BLEDevice1('BC0704B0-D22E-2F3C-56A6-14C044CA42D1')

    await device.connect()
    print("连接成功")
    await device.write(b'\x01')
    await device.notify(callback_new_)
    print("开始监听")
    while True:
        # 异步函数的逻辑
        await asyncio.sleep(1)


def get_time_interval_per_point(len_pulse_wave_data, start_time, end_time):
    total_milliseconds = (end_time - start_time).total_seconds() * 1000
    time_interval_per_point = total_milliseconds / len_pulse_wave_data
    return time_interval_per_point


def get_nn_intervals_ms(pulse_wave_data, start_time, end_time):
    time_interval_per_point = get_time_interval_per_point(len(pulse_wave_data), start_time, end_time)

    # 定义一个阈值，用于确定脉搏波峰值
    threshold = -10 # 0

    # 寻找脉搏波峰值对应的时间点
    peak_indices = []
    for i in range(1, len(pulse_wave_data) - 1):
        if pulse_wave_data[i] > pulse_wave_data[i - 1] and pulse_wave_data[i] > pulse_wave_data[i + 1] and \
                pulse_wave_data[i] > threshold:
            peak_indices.append(i)

    # 计算相邻两个脉搏波峰值之间的时间间隔，即NN间隔（以毫秒为单位）
    nn_intervals_ms = []
    for i in range(1, len(peak_indices)):
        time_interval_ms = (peak_indices[i] - peak_indices[i - 1]) * (
                    time_interval_per_point)  # 1.5秒采集64个数据，因此每个数据点对应的时间间隔为1.5/64秒，再乘以1000转换为毫秒
        nn_intervals_ms.append(time_interval_ms)

    return nn_intervals_ms


def gen_data(pulses):
    nn_intervals = get_nn_intervals_ms(pulses)

    # 1. SDNN：全部正常窦性心搏间期（NN）的标准差，单位：ms。
    sdnn = np.std(nn_intervals)

    # 2. SDANN：全程按5分钟分成连续的时间段，先计算每5分钟的NN间期平均值，再计算所有平均值的标准差，单位：ms。
    # 这里我们假设数据是连续的，所以我们将使用相邻间隔平均值
    nn_intervals_mean_5min = [np.mean(nn_intervals[i:i + 300]) for i in range(0, len(nn_intervals), 300)]
    sdann = np.std(nn_intervals_mean_5min)

    # 3. RMSSD：全程相邻NN间期之差的均方根值，单位：ms。
    rmssd = np.sqrt(np.mean(np.square(np.diff(nn_intervals))))

    # 4. SDNN Index：全程按5分钟分成连续的时间段，先计算每5分钟的NN间期标准差，再计算这些标准差的平均值，单位：ms。
    sdnn_index = np.mean([np.std(nn_intervals[i:i + 300]) for i in range(0, len(nn_intervals), 300)])

    # 5. SDSD：全部相邻NN间期之差的标准差，单位：ms。
    sdsd = np.std(np.diff(nn_intervals))

    # 6. NN50：全部NN间期中，相邻的NN间期之差大于50ms的心搏数，单位：个。
    nn50 = np.sum(np.abs(np.diff(nn_intervals)) > 50)

    # 7. PNN50：NN50除以总的NN间期个数，乘以100，单位：%。
    pnn50 = (nn50 / len(nn_intervals)) * 100

    # 打印计算结果
    print("1. SDNN:", sdnn, "ms")
    print("2. SDANN:", sdann, "ms")
    print("3. RMSSD:", rmssd, "ms")
    print("4. SDNN Index:", sdnn_index, "ms")
    print("5. SDSD:", sdsd, "ms")
    print("6. NN50:", nn50)
    print("7. PNN50:", pnn50, "%")


def calculate_sdnn(nn_intervals):
    # 将心跳间隔数据转换为numpy数组
    nn_intervals = np.array(nn_intervals)

    # 计算标准差NN间隔（SDNN）
    sdnn = np.std(nn_intervals)

    return sdnn


def calculate_rmssd(nn_intervals):
    # 计算相邻RR间隔的差异
    nn_diff = np.diff(nn_intervals)

    # 计算差异的平方
    nn_diff_squared = nn_diff ** 2

    # 计算均方根差（RMSSD）
    rmssd = np.sqrt(np.mean(nn_diff_squared))

    return rmssd


def calculate_pnn50(nn_intervals):
    # 计算相邻RR间隔的差异
    nn_diff = [abs(nn_intervals[i + 1] - nn_intervals[i]) for i in range(len(nn_intervals) - 1)]

    # 计算差异大于50毫秒的次数
    nn_diff_gt_50ms = sum(diff > 50 for diff in nn_diff)

    # 计算PNN50指标
    pnn50 = (nn_diff_gt_50ms / (len(nn_intervals) - 1)) * 100

    return pnn50


def calculate_cv(nn_intervals):
    # 将心跳间隔数据转换为numpy数组
    nn_intervals = np.array(nn_intervals)

    # 计算平均值和标准差
    mean_nn = np.mean(nn_intervals)
    std_nn = np.std(nn_intervals)

    # 计算CV指标
    cv = (std_nn / mean_nn) * 100

    return cv


def calculate_total_power(nn_intervals, fs=4):
    # 使用Welch方法进行频谱密度估计
    f, psd = welch(nn_intervals, fs=fs, nperseg=len(nn_intervals))

    # 计算总功率（在频段VLF到HF内）
    vlf_indices = np.where((f >= 0.0033) & (f < 0.04))  # 非常低频(VLF)范围
    lf_indices = np.where((f >= 0.04) & (f < 0.15))  # 低频(LF)范围
    hf_indices = np.where((f >= 0.15) & (f <= 0.4))  # 高频(HF)范围

    total_power = np.trapz(psd[vlf_indices]) + np.trapz(psd[lf_indices]) + np.trapz(psd[hf_indices])

    return total_power


def calculate_ulf_power(nn_intervals, fs=4):
    # 使用Welch方法进行频谱密度估计
    f, psd = welch(nn_intervals, fs=fs, nperseg=len(nn_intervals))

    # 计算ULF功率（在频段ULF内）
    ulf_indices = np.where((f >= 0.0033) & (f < 0.04))  # 超低频(ULF)范围
    ulf_power = np.trapz(psd[ulf_indices])

    return ulf_power


def calculate_vlf_power(nn_intervals, fs=4):
    # 使用Welch方法进行频谱密度估计
    f, psd = welch(nn_intervals, fs=fs, nperseg=len(nn_intervals))

    # 计算VLF功率（在频段VLF内）
    vlf_indices = np.where((f >= 0.0033) & (f < 0.04))  # 非常低频(VLF)范围
    vlf_power = np.trapz(psd[vlf_indices])

    return vlf_power


def calculate_lf_power(nn_intervals, fs=4):
    # 使用Welch方法进行频谱密度估计
    f, psd = welch(nn_intervals, fs=fs, nperseg=len(nn_intervals))

    # 计算LF功率（在频段LF内）
    lf_indices = np.where((f >= 0.04) & (f < 0.15))  # 低频(LF)范围
    lf_power = np.trapz(psd[lf_indices])

    return lf_power


def calculate_hf_power(nn_intervals, fs=4):
    # 使用Welch方法进行频谱密度估计
    f, psd = welch(nn_intervals, fs=fs, nperseg=len(nn_intervals))

    # 计算HF功率（在频段HF内）
    hf_indices = np.where((f >= 0.15) & (f <= 0.4))  # 高频(HF)范围
    hf_power = np.trapz(psd[hf_indices])

    return hf_power


def calculate_lfnorm(nn_intervals, fs=4):
    # 计算LF功率
    lf_power = calculate_lf_power(nn_intervals, fs)

    # 计算总功率
    total_power = calculate_total_power(nn_intervals, fs)

    # 计算LFnorm
    lfnorm = lf_power / total_power

    return lfnorm


def calculate_hfnorm(nn_intervals, fs=4):
    # 计算HF功率
    hf_power = calculate_hf_power(nn_intervals, fs)

    # 计算总功率
    total_power = calculate_total_power(nn_intervals, fs)

    # 计算HFnorm
    hfnorm = hf_power / total_power

    return hfnorm


def calculate_dc_component(nn_intervals):
    # 计算心率变异性信号的均值（直流分量）
    dc_component = np.mean(nn_intervals)

    return dc_component


def get_data_metrics(pulse_wave_data, start_time, end_time):
    nn_intervals = get_nn_intervals_ms(pulse_wave_data, start_time, end_time)
    print(nn_intervals)
    return {
        'sdnn': calculate_sdnn(nn_intervals),
        'rmssd': calculate_rmssd(nn_intervals),
        'pnn50': calculate_pnn50(nn_intervals),
        'lf': calculate_lf_power(nn_intervals),
        'hf': calculate_hf_power(nn_intervals),
        'lfnorm': calculate_lfnorm(nn_intervals),
        'hfnorm': calculate_hfnorm(nn_intervals)
    }


def get_heart_rate_variability_scatter_plot(pulse_wave_data, start_time, end_time):
    time_interval = get_time_interval_per_point(len(pulse_wave_data), start_time, end_time)

    threshold = 0
    # 识别峰值位置
    peaks = [i for i in range(1, len(pulse_wave_data) - 1) if pulse_wave_data[i] > threshold and pulse_wave_data[i - 1] < pulse_wave_data[i] > pulse_wave_data[i + 1]]

    # 计算RR间期
    RR_intervals = [(peaks[i + 1] - peaks[i]) * time_interval for i in range(len(peaks) - 1)]

    # 构建散点图数据
    x_data = RR_intervals[:-1]
    y_data = RR_intervals[1:]
    return {
        'x_data': x_data,
        'y_data': y_data
    }


if __name__ == "__main__":
    def callback_(data: dict):
        print(data)
    asyncio.run(receive_heart_rate(callback_))
