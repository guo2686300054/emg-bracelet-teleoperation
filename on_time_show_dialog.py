import queue
import sys
import threading
import time
import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QColor
from pyqtgraph import PlotWidget
from UI_on_time_show_dialog import Ui_Dialog


class OnTimeShowDialog(QDialog, Ui_Dialog):
    wg_display_signal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, wg_repeat_time=21, wg_channel=5):
        super(OnTimeShowDialog, self).__init__()
        self.setupUi(self)
        # 修复窗口标志 - 添加最小化按钮标志
        self.setWindowFlags(self.windowFlags() |
                            QtCore.Qt.WindowMaximizeButtonHint |
                            QtCore.Qt.WindowMinimizeButtonHint |
                            QtCore.Qt.WindowSystemMenuHint)  # 关键修复

        self.listWidget.setParent(self)
        # 修改子窗口标志为普通控件
        self.listWidget.setWindowFlags(QtCore.Qt.Widget)  # 关键修复
        self.listWidget.setGeometry(QtCore.QRect(0, 0, 111, 141))
        # 画笔颜色设置
        self.pens = ("k", "grey", "red", "peru", "gold", "green", "cyan", "blue")

        # 信号连接
        self.wg_display_signal.connect(self.wg_channels_display)

        # 数据处理相关属性
        self.wave_guide_queue = queue.Queue(1000)
        self.wave_last_channel = []
        self.wg_channels = wg_channel
        self.wg_repeat_time = wg_repeat_time
        self.wg_received_data_len = 0
        self.wg_show_len = 200

        # 通道选择相关属性
        self.displayed_channels = list(range(wg_channel))  # 默认显示所有通道

        # 初始化绘图
        self.wg_plot = None
        self.wg_init_plot()

        # 初始化通道选择列表
        self.init_channel_list()

        # 使用 QTimer 替代手动线程管理
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.onTimer)
        self.timer.start(20)  # 20ms，即50Hz

    def resizeEvent(self, event):
        """处理窗口大小变化事件"""
        super(OnTimeShowDialog, self).resizeEvent(event)
        self.update_list_widget_position()

    def update_list_widget_position(self):
        """更新列表控件到右上角位置"""
        if hasattr(self, 'listWidget') and self.listWidget.isVisible():
            # 计算右上角位置（留出10像素边距）
            x_pos = self.width() - self.listWidget.width() - 10
            y_pos = 10

            # 设置新位置
            self.listWidget.move(x_pos, y_pos)

    def showEvent(self, event):
        """窗口显示时更新位置"""
        super(OnTimeShowDialog, self).showEvent(event)
        self.update_list_widget_position()
        # 确保窗口正常显示
        self.activateWindow()
        self.raise_()
    def init_channel_list(self):
        """初始化通道选择列表"""
        # 连接项目变化信号
        self.listWidget.itemChanged.connect(self.channel_selection_changed)

        # 确保列表项数量与通道数匹配
        while self.listWidget.count() > self.wg_channels + 1:  # +1 为"所有通道"
            self.listWidget.takeItem(self.listWidget.count() - 1)

        # 更新通道名称
        for i in range(self.listWidget.count()):
            item = self.listWidget.item(i)
            if i == 0:
                item.setText("所有通道")
            else:
                item.setText(f"通道{i}")

    def channel_selection_changed(self, item):
        """处理通道选择变化"""
        # 临时断开信号防止递归触发
        self.listWidget.itemChanged.disconnect(self.channel_selection_changed)

        try:
            # 获取所有选中的项
            selected_items = []
            for i in range(self.listWidget.count()):
                if self.listWidget.item(i).checkState() == QtCore.Qt.Checked:
                    selected_items.append(self.listWidget.item(i).text())

            # 判断是否是"所有通道"被点击
            if item.text() == "所有通道":
                if item.checkState() == QtCore.Qt.Checked:
                    # 勾选所有子通道
                    for i in range(1, self.listWidget.count()):
                        self.listWidget.item(i).setCheckState(QtCore.Qt.Checked)
                else:
                    # 取消所有子通道
                    for i in range(1, self.listWidget.count()):
                        self.listWidget.item(i).setCheckState(QtCore.Qt.Unchecked)
            else:
                # 处理子通道勾选变化
                all_checked = True
                for i in range(1, self.listWidget.count()):
                    if self.listWidget.item(i).checkState() != QtCore.Qt.Checked:
                        all_checked = False
                        break

                # 更新"所有通道"状态
                all_channels_item = self.listWidget.item(0)
                if all_checked:
                    all_channels_item.setCheckState(QtCore.Qt.Checked)
                else:
                    all_channels_item.setCheckState(QtCore.Qt.Unchecked)

            # 重新获取选中的通道
            selected_items = []
            for i in range(self.listWidget.count()):
                if self.listWidget.item(i).checkState() == QtCore.Qt.Checked:
                    selected_items.append(self.listWidget.item(i).text())

            # 确定显示哪些通道
            if "所有通道" in selected_items:
                self.displayed_channels = list(range(self.wg_channels))
            else:
                self.displayed_channels = []
                for item_text in selected_items:
                    if item_text.startswith("通道"):
                        channel_num = int(item_text[2:])  # 提取通道号
                        if channel_num - 1 < self.wg_channels:
                            self.displayed_channels.append(channel_num - 1)

            # 更新通道可见性
            self.update_channel_visibility()

            # 如果有数据存在，重新更新显示
            if hasattr(self, 'last_displayed_data'):
                self.wg_display_signal.emit(self.last_displayed_data)

        finally:
            # 重新连接信号
            self.listWidget.itemChanged.connect(self.channel_selection_changed)

    def update_channel_visibility(self):
        """根据当前选择的通道更新曲线的可见性"""
        if not hasattr(self, 'wg_plot'):
            return

        # 遍历所有曲线对象
        for channel_idx, plot in enumerate(self.wg_plot):
            # 如果当前通道在显示列表中
            if channel_idx in self.displayed_channels:
                plot.show()  # 显示该通道曲线
            else:
                plot.hide()  # 隐藏该通道曲线

    def receive_waveguide_signal(self, waveguide_data):
        """接收光波导数据信号槽"""
        if not self.wave_guide_queue.full():
            self.wave_guide_queue.put_nowait(waveguide_data)

    def wg_init_plot(self):
        """光波导画布初始化"""
        # 设置背景为白色
        self.wave_guide_show.setBackground('w')

        # 设置坐标轴颜色为黑色
        self.wave_guide_show.getAxis('bottom').setPen('k')
        self.wave_guide_show.getAxis('left').setPen('k')

        self.wg_plot = []

        # 创建所有通道的曲线（初始都可见）
        for i in range(0, self.wg_channels):
            plot = self.wave_guide_show.plot(pen=self.pens[i], name=f"Channel {i + 1}")
            self.wg_plot.append(plot)

        # 设置Y轴范围为0-300
        self.wave_guide_show.setYRange(0, 300)

    def wg_channels_display(self, data):
        """根据选择的通道显示数据"""
        # 保存当前数据，以便通道切换时重绘
        self.last_displayed_data = data

        x = np.arange(0, data.shape[0])

        # 只更新需要显示的通道
        for channel_idx in self.displayed_channels:
            self.wg_plot[channel_idx].setData(x, data[:, channel_idx])

    def onTimer(self):
        """定时器槽函数，用于定期从队列中获取数据并更新显示"""
        while not self.wave_guide_queue.empty():
            data = self.wave_guide_queue.get()
            data = np.asarray(data)  # 如果确定 data 已经是 ndarray，可以省略
            new_data = self.get_full_channel_wg_data(data)
            self.wg_display_signal.emit(new_data)

    def get_full_channel_wg_data(self, wg_data):
        """获得所有通道的光波导数据并重塑"""
        new_data = wg_data.reshape(-1, self.wg_channels)
        new_data = wg_data.reshape(-1, self.wg_channels)


        new_data = np.clip(new_data, 0, 255)

        if self.wg_received_data_len == 0:
            self.wave_last_channel = new_data
            self.wg_received_data_len += 1
        elif self.wg_received_data_len < self.wg_show_len:
            self.wave_last_channel = np.concatenate((self.wave_last_channel, new_data), axis=0)
            self.wg_received_data_len += 1
        elif self.wg_received_data_len == self.wg_show_len:
            tem_data = self.wave_last_channel[new_data.shape[0]:, :]
            self.wave_last_channel = np.concatenate((tem_data, new_data), axis=0)

        # 只保留最新的 wg_show_len 数据点
        # if self.wave_last_channel.shape[0] > self.wg_show_len:
        #     self.wave_last_channel = self.wave_last_channel[-self.wg_show_len:, :]

        return self.wave_last_channel

    def resetLastOneChannel(self):
        """重置数据缓冲区"""
        self.wave_last_channel = []
        self.wg_received_data_len = 0
        if hasattr(self, 'last_displayed_data'):
            del self.last_displayed_data


class MultiChannelShowDialog(QDialog):
    wg_display_signal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, wg_repeat_time=21, wg_channel=5):
        super(MultiChannelShowDialog, self).__init__()
        # 修复窗口标志 - 添加最小化按钮标志
        self.setWindowFlags(self.windowFlags() |
                            QtCore.Qt.WindowMaximizeButtonHint |
                            QtCore.Qt.WindowMinimizeButtonHint |
                            QtCore.Qt.WindowSystemMenuHint)  # 关键修复

        # 创建主布局
        self.main_layout = QHBoxLayout(self)

        # 创建左侧区域（滚动区域+通道选择）
        self.left_layout = QVBoxLayout()
        self.main_layout.addLayout(self.left_layout)

        # 创建滚动区域
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QtWidgets.QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setAlignment(QtCore.Qt.AlignTop)
        self.scroll_area.setWidget(self.scroll_content)
        self.left_layout.addWidget(self.scroll_area)

        # 画笔颜色设置
        self.pens = ("k", "grey", "red", "peru", "gold", "green", "cyan", "blue")

        # 信号连接
        self.wg_display_signal.connect(self.wg_channels_display)

        # 数据处理相关属性
        self.wave_guide_queue = queue.Queue(1000)
        self.wave_last_channel = []
        self.wg_channels = wg_channel
        self.wg_repeat_time = wg_repeat_time
        self.wg_received_data_len = 0
        self.wg_show_len = 200

        # 通道选择相关属性
        self.displayed_channels = list(range(wg_channel))  # 默认显示所有通道

        # 初始化绘图
        self.plot_widgets = []  # 存储每个通道的PlotWidget
        self.plot_curves = []  # 存储每个通道的曲线对象
        self.wg_init_plot()

        # 使用 QTimer 替代手动线程管理
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.onTimer)
        self.timer.start(20)  # 20ms，即50Hz

    def showEvent(self, event):
        """窗口显示时确保正常激活"""
        super(MultiChannelShowDialog, self).showEvent(event)
        self.activateWindow()
        self.raise_()
    def receive_waveguide_signal(self, waveguide_data):
        """接收光波导数据信号槽"""
        if not self.wave_guide_queue.full():
            self.wave_guide_queue.put_nowait(waveguide_data)

    def wg_init_plot(self):
        """光波导画布初始化 - 为每个通道单独初始化"""
        # 清除滚动区域内容
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        # 清空列表
        self.plot_widgets.clear()
        self.plot_curves.clear()

        # 创建主网格布局
        grid_widget = QtWidgets.QWidget()
        grid_layout = QtWidgets.QGridLayout(grid_widget)
        grid_layout.setSpacing(10)  # 设置通道间的间距

        # 将网格布局添加到滚动区域
        self.scroll_layout.addWidget(grid_widget)

        # 固定每个通道的高度
        channel_height = 150  # 每个通道固定高度

        # 计算每行显示的通道数（2列）
        channels_per_row = 2

        for i in range(self.wg_channels):
            # 创建单个通道的容器
            channel_container = QtWidgets.QWidget()
            channel_container.setFixedHeight(channel_height)
            channel_layout = QVBoxLayout(channel_container)
            channel_layout.setContentsMargins(5, 5, 5, 5)
            channel_layout.setSpacing(5)

            # 创建通道标题
            channel_label = QtWidgets.QLabel(f"通道 {i + 1}")
            channel_label.setStyleSheet("font-weight: bold; color: #333; font-size: 12px;")
            channel_layout.addWidget(channel_label)

            # 创建绘图区域
            plot = PlotWidget()
            plot.setBackground('w')
            plot.getAxis('bottom').setPen('k')
            plot.getAxis('left').setPen('k')
            plot.setYRange(0, 300)

            # 创建曲线对象
            curve = plot.plot(pen=self.pens[i], name=f"通道 {i + 1}")
            self.plot_curves.append(curve)
            self.plot_widgets.append(plot)

            # 添加到通道容器
            channel_layout.addWidget(plot)

            # 计算在网格中的位置
            row = i // channels_per_row
            col = i % channels_per_row

            # 将通道容器添加到网格布局
            grid_layout.addWidget(channel_container, row, col)

        # 移除滚动区域大小调整（不再需要）
        # self.adjust_scroll_area_size()

    def wg_channels_display(self, data):
        """根据选择的通道显示数据"""
        # 保存当前数据，以便通道切换时重绘
        self.last_displayed_data = data
        x = np.arange(0, data.shape[0])

        # 更新每个通道的显示
        for channel_idx in range(self.wg_channels):
            # 只有当通道在显示列表中时才更新
            if channel_idx in self.displayed_channels:
                # 使用 plot_curves 而不是 wg_plots
                self.plot_curves[channel_idx].setData(x, data[:, channel_idx])

    def onTimer(self):
        """定时器槽函数，用于定期从队列中获取数据并更新显示"""
        while not self.wave_guide_queue.empty():
            data = self.wave_guide_queue.get()
            data = np.asarray(data)  # 如果确定 data 已经是 ndarray，可以省略
            new_data = self.get_full_channel_wg_data(data)
            self.wg_display_signal.emit(new_data)

    def get_full_channel_wg_data(self, wg_data):
        """获得所有通道的光波导数据并重塑"""
        new_data = wg_data.reshape(-1, self.wg_channels)
        new_data = np.clip(new_data, 0, 255)

        if self.wg_received_data_len == 0:
            self.wave_last_channel = new_data
            self.wg_received_data_len += 1
        elif self.wg_received_data_len < self.wg_show_len:
            self.wave_last_channel = np.concatenate((self.wave_last_channel, new_data), axis=0)
            self.wg_received_data_len += 1
        elif self.wg_received_data_len == self.wg_show_len:
            tem_data = self.wave_last_channel[new_data.shape[0]:, :]
            self.wave_last_channel = np.concatenate((tem_data, new_data), axis=0)

        return self.wave_last_channel

    def resetLastOneChannel(self):
        """重置数据缓冲区"""
        self.wave_last_channel = []
        self.wg_received_data_len = 0
        if hasattr(self, 'last_displayed_data'):
            del self.last_displayed_data

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    form = OnTimeShowDialog(wg_channel=5)
    form.show()


    # 模拟数据输入
    def generate_test_data():
        """生成测试数据"""
        while True:
            # 生成随机波形数据 (5通道 × 21点)
            data = np.random.rand(5 * 21) * 5 + np.sin(np.linspace(0, 4 * np.pi, 5 * 21))
            form.receive_waveguide_signal(data)
            time.sleep(0.02)  # 20Hz数据输入频率


    # 启动数据生成线程
    threading.Thread(target=generate_test_data, daemon=True).start()

    sys.exit(app.exec_())