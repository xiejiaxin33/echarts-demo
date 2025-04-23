import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # 示例数据
    time = np.arange(0, 100, 1)
    heart_rate = np.random.normal(70, 10, size=100)

    # 绘制散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(time, heart_rate, color='blue', alpha=0.5)

    # 添加标题和标签
    plt.title('Heart Rate Scatter Plot')
    plt.xlabel('Time (s)')
    plt.ylabel('Heart Rate (bpm)')
    plt.grid(True)
    plt.savefig('heart_rate_scatter_plot.png')
    plt.show()
