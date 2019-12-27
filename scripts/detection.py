import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import QTimer
from scripts import config

if config.DEBUG:
    from scripts.wood_detection import Ui_dialog
    from scripts.parameter import Ui_Dialog_Child
    from scripts.notes import Ui_Dialog_Notes
    from scripts.utils import new_show_result, show_result, new_show_result_2, new_show_result_3, creat_match_graph_txt, \
        get_result_from_meger_file
    from scripts.utils import new_show_result_21
    from scripts.merger_result import get_final_result, get_final_result_multirows
    import scripts.utils

else:
    from scripts.wood_detection import Ui_dialog
    from scripts.parameter import Ui_Dialog_Child
    from scripts.notes import Ui_Dialog_Notes
    from scripts.utils import new_show_result, show_result, new_show_result_2,new_show_result_3, creat_match_graph_txt, get_result_from_meger_file
    from scripts.utils import  new_show_result_21
    # from scripts.merger_result import get_final_result, get_final_result_multirows
    from scripts.merger_result_v2 import  get_final_result,get_final_result_multirows
    import scripts.utils

from PIL import Image
import cv2
import numpy as np
import threading
import time
import shutil
from scripts.softp import SoftwareProtecttion
import pyrealsense2 as pr

#继承原始的设计类，添加相关的控件回调函数
#方便修改前面界面和后面回调函数，界面布局原始类发生变化只要直接替换wood_detection_ui.py就行
#需要确定每个控件的object_name，保持一致性.这里采用了默认的命名方式
#######################################################
# 图像控件：pushButton
# 检测控件：pushButton_2
# 暂停控件：pushButton_3
# 原始图窗控件：graphicsView
# 检测图窗控件：graphicsView_2

class Detection_Notes(QDialog, Ui_Dialog_Notes):
    def __init__(self):
        super(Detection_Notes, self).__init__()
        self.setupUi(self)

class Detection_Child(QDialog, Ui_Dialog_Child):
    def __init__(self):
        super(Detection_Child, self).__init__()
        self.setupUi(self)
        self.checkBox.setChecked(True)
        self.horizontalSlider_9.setEnabled(False)
        self.checkBox.clicked.connect(self.adjust)
        self.pushButton.clicked.connect(self.reset)
        self.expos = True
        self.expos_init =  [0, 50, 64, 300, 0, 64, 50, 4600]
        self.reset_flag = False

    def adjust(self):
        if self.checkBox.isChecked() == False:
            self.expos = False
            self.horizontalSlider_9.setEnabled(True)
        else:
            self.expos = True
            self.horizontalSlider_9.setEnabled(False)

    def update(self):
        if self.reset_flag == True:
            self.reset_flag = False
            return self.expos_init
        brightness = self.horizontalSlider.value()
        contrast = self.horizontalSlider_2.value()
        grain = self.horizontalSlider_3.value()
        gamma = self.horizontalSlider_4.value()
        hue = self.horizontalSlider_5.value()
        staturation = self.horizontalSlider_7.value()
        sharpness = self.horizontalSlider_8.value()
        white_balance = self.horizontalSlider_6.value()

        results = [brightness, contrast, grain, gamma, hue, staturation, sharpness, white_balance]
        return results

    def exposfun(self):
        if self.reset_flag == True:
            self.expos = True
            return 'reset'
        expos = 0
        if self.expos == False:
            expos = self.horizontalSlider_9.value()
        else:
            return 'auto'
        return expos

    def reset(self):
        self.horizontalSlider.setValue(self.expos_init[0])
        self.horizontalSlider_2.setValue(self.expos_init[1])
        self.horizontalSlider_3.setValue(self.expos_init[2])
        self.horizontalSlider_4.setValue(self.expos_init[3])
        self.horizontalSlider_5.setValue(self.expos_init[4])
        self.horizontalSlider_7.setValue(self.expos_init[5])
        self.horizontalSlider_8.setValue(self.expos_init[6])
        self.horizontalSlider_6.setValue(self.expos_init[7])
        self.horizontalSlider_9.setEnabled(False)
        self.checkBox.setChecked(True)
        self.reset_flag = True


class Detection_Ui(QtWidgets.QMainWindow,Ui_dialog):
    def __init__(self):
        super(Detection_Ui, self).__init__()
        self.setupUi(self)
        #设置控件的回调函数
        # 添加回调函数  后面改为继承
        #software protection
        self.softp = SoftwareProtecttion(elapsed_time=90, elapsed_time_flag=True, mac_protection_flag=False)
        self.child = Detection_Child()
        self.notes = Detection_Notes()
        self.timer = QTimer()
        self.org_img = None
        self.img_list = []
        self.img_list_raw = []
        self.isimage = True
        self.isvideo = False
        self.ismulti = False
        self.issaveimage = False
        self.savevideo_flag = False
        self.save_woods_nums = 1
        self.image_num = 0
        self.video_reset = False
        self.paly_terminate_flag = False
        self.paly_reset_flag = False
        self.widget.show()
        self.widget_2.hide()
        self.pushButton.clicked.connect(self.open_file_and_show_img)
        self.pushButton_2.clicked.connect(self.detect)
        self.pushButton_3.clicked.connect(self.saveimage)
        self.pushButton_4.clicked.connect(self.videoplay)
        self.pushButton_5.clicked.connect(self.videoreset)
        self.pushButton_6.clicked.connect(self.videoterminate)
        self.pushButton_7.clicked.connect(self.parameteradjust)
        self.adjust = False
        self.expos_init = [0, 50, 64, 300, 0, 64, 50, 4600]
        self.timer.timeout.connect(self.close_win)
        self.radioButton.clicked.connect(self.radioButtonimage)
        self.radioButton_2.clicked.connect(self.radioButtonvideo)
        self.radioButton_3.clicked.connect(self.radioButtonmulti)
        self.radioButton_4.clicked.connect(self.radioButtonSaveImage)
        self.stopEvent = threading.Event()
        self.stopEvent.clear()
        self.radioButton.setChecked(True)
        self.lineEdit.setReadOnly(True)
        self.config_file = './scripts/config.py'
        self.class_names = ['Wood', 'Wood']
        self.checkpoint_file_list = ['model.pth', 'epoch_1000.pth', 'epoch_1500.pth']

        self.pause_det = False
        self.time = 0
        self.now = ""

    def closeEvent(self, event):
        """
        对MainWindow的函数closeEvent进行重构
        退出软件时结束所有进程
        :param event:
        :return:
        """
        reply = QtWidgets.QMessageBox.question(self,
                                               '本程序',
                                               "是否要退出程序？",
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
            #print(self.issaveimage)
            sys.exit(0)
        else:
            event.ignore()

    def close_win(self):
        """
        重写closeEvent方法，实现dialog窗体关闭时执行一些代码
        :param event: close()触发的事件
        :return: None
        """
        #reply = QMessageBox.Warning(self, '本程序',
        #                                       "你的软件使用时间或mac地址异常，将强制退出？")
        # if reply == QtWidgets.QMessageBox.Yes:
        #     QMainWindow.close()
        # else:
        #     QMainWindow.close()
        self.textEdit.setText("软件使用时间或mac地址异常，将在10后强制退出!")
        QApplication.processEvents()
        time.sleep(10)
        sys.exit(0)
    #暂停信号的控制
    def detect_command(self):
        if(self.pause_det==False):
            self.pause_det = True
            self.pushButton_3.setText("继续")
        else:
            self.pause_det= True
            self.pushButton_3.setText("暂停")

    # 在第二个图窗中显示结果并且在text显示检测结果的相关信息
    def radioButtonimage(self):
        self.graphicsView.setScene(None)
        self.graphicsView_2.setScene(None)
        self.graphicsView_3.setScene(None)
        self.textEdit.setText("开始单图测试！" + '\n')
        QApplication.processEvents()
        self.isimage = True
        self.isvideo = False
        self.ismulti = False
        self.issaveimage = False
        self.lineEdit.setReadOnly(True)
        self.lineEdit_2.setReadOnly(True)
        self.stopEvent.set()
        self.paly_reset_flag = True
        self.widget.show()
        self.widget_2.hide()


    def radioButtonvideo(self):
        self.graphicsView.setScene(None)
        self.graphicsView_2.setScene(None)
        self.graphicsView_3.setScene(None)
        self.textEdit.setText("开始视频测试！" + '\n')
        QApplication.processEvents()
        self.isvideo = True
        self.isimage = False
        self.ismulti = False
        self.issaveimage = False

        self.lineEdit.setReadOnly(True)
        self.lineEdit_2.setReadOnly(True)
        self.paly_reset_flag = True
        self.widget.show()
        self.widget_2.hide()

    def radioButtonmulti(self):
        self.graphicsView.setScene(None)
        self.graphicsView_2.setScene(None)
        self.graphicsView_3.setScene(None)
        self.textEdit.setText("开始多图测试！" + '\n')
        QApplication.processEvents()
        self.lineEdit.setReadOnly(False)
        self.lineEdit_2.setReadOnly(True)
        self.isvideo = False
        self.isimage = False
        self.ismulti = True
        self.issaveimage = False

        self.stopEvent.set()
        self.paly_reset_flag = True
        self.widget.hide()
        self.widget_2.show()

    def radioButtonSaveImage(self):
        self.textEdit.setText("请保证摄像头已经插入！" + '\n')
        self.lineEdit.setReadOnly(True)
        self.isvideo = False
        self.isimage = False
        self.ismulti = False
        self.issaveimage = True
        self.paly_reset_flag = False
        self.stopEvent.set()
        self.widget.show()
        self.widget_2.hide()
        self.lineEdit_2.setReadOnly(False)
        self.cur_acquire_flag = 0

    def parameteradjust(self):

        self.adjust = True
        self.child.show()

    def videoplay(self):
        if self.issaveimage:
            self.pushButton_3.setEnabled(True)
            self.notes.show()
            self.notes.textBrowser.setText('\n')
            self.notes.textBrowser.append('\t\t\t\t采集一排' + '\n')
            self.notes.textBrowser.append('  采集一排请按木头堆从左至右的顺序拍摄' + '\n')
            self.notes.textBrowser.append('  请保证一个木头堆的拍摄角度基本一致，尽量在一个水平线上移动拍摄，且保证相邻帧的重叠面积在50%左右，且拍摄清晰没有模糊' + '\n')
            self.notes.textBrowser.append('  具体的顺序如下： \n ')
            self.notes.textBrowser.append('\t\t一排：1.png -> 2.png -> 3.png -> 4.png -> 5.png' + '\n')
            self.notes.textBrowser.append('\t\t\t\t采集两排 \n')
            self.notes.textBrowser.append('  采集二排请按木头堆指定从左至右，先下再上，再下再上顺序拍摄' + '\n')
            self.notes.textBrowser.append('  并保证一个木头堆的拍摄角度基本一致，尽量在一个水平线上移动拍摄，且保证相邻帧的重叠面积在50%左右，且拍摄清晰没有模糊' + '\n')
            self.notes.textBrowser.append('  具体的顺序如下： \n')
            self.notes.textBrowser.append(
                '\t\t二排：2.png -> 4.png -> 6.png -> 8.png -> 10.png' + '\n' + '\t\t一排：1.png -> 3.png -> 5.png -> 7.png -> 9.png' + '\n')
            QApplication.processEvents()
            self.paly_reset_flag = False
            save_folders = './save/imgs'
            # if os.path.exists(save_folders):
            #     shutil.rmtree(save_folders)
            if not os.path.exists(save_folders):
                os.makedirs(save_folders)
            self.now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
            pl = pr.pipeline()
            con = pr.config()

            con.enable_stream(pr.stream.depth, 640, 480, pr.format.z16, 15)
            con.enable_stream(pr.stream.color, 640, 480, pr.format.bgr8, 15)
            # Start streaming
            pl.start(con)
            sensor = pl.get_active_profile().get_device().query_sensors()[1]
            sensor.set_option(pr.option.enable_auto_exposure, 1)
            # print('herehere')
            sensor.set_option(pr.option.enable_auto_exposure, 1)
            expos = 0
            sensor.set_option(pr.option.brightness, self.expos_init[0])
            sensor.set_option(pr.option.contrast, self.expos_init[1])
            sensor.set_option(pr.option.gain, self.expos_init[2])
            sensor.set_option(pr.option.gamma, self.expos_init[3])
            sensor.set_option(pr.option.hue, self.expos_init[4])
            sensor.set_option(pr.option.saturation, self.expos_init[5])
            sensor.set_option(pr.option.sharpness, self.expos_init[6])
            sensor.set_option(pr.option.white_balance, self.expos_init[7])
            while True:
                sticher_flag = self.lineEdit_2.text()
                if self.paly_reset_flag == True:
                    self.lineEdit_2.setEnabled(True)
                    if sticher_flag is "":
                        self.save_woods_nums = 1
                        self.textEdit.setText('需采集排数值为空，必须为大于0的整数，请输入正确的值！')
                        QApplication.processEvents()
                    else:
                        if int(sticher_flag) <= 0 or str(int(sticher_flag)) != sticher_flag or int(sticher_flag) > 2:
                            self.save_woods_nums = 1
                            self.textEdit.setText('需采集排数值有误，必须为大于0小于3的整数，请重新输入！')
                            QApplication.processEvents()
                        else:

                            self.save_woods_nums += 1
                            self.textEdit.setText('开始采集第' + str(self.save_woods_nums) + '个木头堆！！' + '\n')
                    break
                if self.paly_terminate_flag == False:
                    if self.adjust == True:
                        # self.adjust = False
                        expos = self.child.exposfun()
                        results = self.child.update()
                        if expos == 'auto':
                            # Set the exposure anytime during the operation
                            sensor.set_option(pr.option.enable_auto_exposure, 1)
                        elif expos == 'reset':
                            sensor.set_option(pr.option.enable_auto_exposure, 1)
                        else:
                            sensor.set_option(pr.option.enable_auto_exposure, 0)
                            sensor.set_option(pr.option.exposure, int(expos))

                        brightness = results[0]
                        contrast = results[1]
                        gain = results[2]
                        gamma = results[3]
                        hue = results[4]
                        staturation = results[5]
                        sharpness = results[6]
                        white_balance = results[7]
                        # print('brightness', brightness)
                        if expos == 'reset':
                            sensor.set_option(pr.option.brightness, brightness)
                            sensor.set_option(pr.option.contrast, contrast)
                            sensor.set_option(pr.option.gain, gain)
                            sensor.set_option(pr.option.gamma, gamma)
                            sensor.set_option(pr.option.hue, hue)
                            sensor.set_option(pr.option.saturation, staturation)
                            sensor.set_option(pr.option.sharpness, sharpness)
                            sensor.set_option(pr.option.white_balance, white_balance)
                        else:
                            if brightness != self.expos_init[0]:
                                sensor.set_option(pr.option.brightness, brightness)
                            if contrast != self.expos_init[1]:
                                sensor.set_option(pr.option.contrast, contrast)
                            if gain != self.expos_init[2]:
                                sensor.set_option(pr.option.gain, gain)
                            if gamma != self.expos_init[3]:
                                sensor.set_option(pr.option.gamma, gamma)
                            if hue != self.expos_init[4]:
                                sensor.set_option(pr.option.hue, hue)
                            if staturation != self.expos_init[5]:
                                sensor.set_option(pr.option.saturation, staturation)
                            if sharpness != self.expos_init[6]:
                                sensor.set_option(pr.option.sharpness, sharpness)
                            if white_balance != self.expos_init[7]:
                                sensor.set_option(pr.option.white_balance, white_balance)
                        self.expos_init = results
                    frames = pl.wait_for_frames()
                    depth_frame = frames.get_depth_frame()
                    color_frame = frames.get_color_frame()
                    if not depth_frame or not color_frame:
                        continue

                    # Convert images to numpy arrays
                    depth_image = np.asanyarray(depth_frame.get_data())
                    color_image = np.asanyarray(color_frame.get_data())

                    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                    if self.savevideo_flag == True:
                        self.lineEdit_2.setEnabled(False)
                        self.savevideo_flag = False

                        if sticher_flag is "":
                            self.image_num = 0
                            self.textEdit.setText('需采集排数值为空，必须为大于0的整数，请输入正确的值！')
                            QApplication.processEvents()
                        else:
                            if int(sticher_flag) <= 0 or str(int(sticher_flag)) != sticher_flag or int(
                                    sticher_flag) > 2:
                                self.image_num = 0
                                self.textEdit.setText('需采集排数值有误，必须为大于0小于3的整数，请重新输入！')
                                QApplication.processEvents()
                            else:
                                if int(sticher_flag) != self.cur_acquire_flag:
                                    self.image_num = 1
                                    # self.save_woods_nums += 1
                                self.cur_acquire_flag = int(sticher_flag)
                                self.textEdit.setText('请看到保存成功标志后，再继续保存下一张！')
                                self.textEdit.append('当前保存第' + str(self.image_num) + '个木头局部图片')
                                QApplication.processEvents()
                                save_folders_other = save_folders + '/' + sticher_flag + '_line'
                                if not os.path.exists(save_folders_other):
                                    os.makedirs(save_folders_other)
                                save_sub_folders = save_folders_other + '/' + self.now

                                if not os.path.exists(save_sub_folders):
                                    os.makedirs(save_sub_folders)
                                cv2.imwrite(save_sub_folders + '/' + str(self.image_num) + '_depth.tif', depth_image)
                                cv2.imwrite(save_sub_folders + '/' + str(self.image_num) + '.png', color_image)
                                cv2.imwrite(save_sub_folders + '/' + str(self.image_num) + '_depth.png', depth_colormap)
                                self.textEdit.setText('保存成功！')

                    img1 = color_image
                    img2 = depth_colormap

                    if img1 is None:
                        continue
                    if img2 is None:
                        continue
                    img1 = cv2.resize(img1, (540, 510))
                    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                    img2 = cv2.resize(img2, (540, 510))
                    x = img1.shape[1]  # 获取图像大小
                    y = img1.shape[0]
                    self.zoomscale = 1  # 图片放缩尺度
                    frame1 = QImage(img1, x, y, QImage.Format_RGB888)
                    pix1 = QPixmap.fromImage(frame1)
                    self.item1 = QGraphicsPixmapItem(pix1)  # 创建像素图元
                    # self.item.setScale(self.zoomscale)
                    self.scene1 = QGraphicsScene()  # 创建场景
                    self.scene1.addItem(self.item1)

                    frame2 = QImage(img2, x, y, QImage.Format_RGB888)
                    pix2 = QPixmap.fromImage(frame2)
                    self.item2 = QGraphicsPixmapItem(pix2)  # 创建像素图元
                    # self.item.setScale(self.zoomscale)
                    self.scene2 = QGraphicsScene()  # 创建场景
                    self.scene2.addItem(self.item2)

                    self.graphicsView.setScene(self.scene1)
                    self.graphicsView_2.setScene(self.scene2)
                    QApplication.processEvents()
                else:
                    self.graphicsView.setScene(None)
                    self.graphicsView_2.setScene(None)
                    QApplication.processEvents()

            self.graphicsView.setScene(None)
            self.graphicsView_2.setScene(None)
            QApplication.processEvents()

    def videoterminate(self):
        if self.paly_terminate_flag == False:
            self.paly_terminate_flag = True
            self.pushButton_6.setText('继续')
            self.pushButton_4.setEnabled(False)
            self.pushButton_3.setEnabled(False)
            QApplication.processEvents()
        else:
            self.paly_terminate_flag = False
            self.pushButton_6.setText('暂停')
            self.pushButton_4.setEnabled(True)
            self.pushButton_3.setEnabled(True)
            QApplication.processEvents()
            # self.graphicsView.setScene(None)
            # self.graphicsView_2.setScene(None)
            # QApplication.processEvents()

    def saveimage(self):
        self.savevideo_flag = True

        self.image_num += 1

    def videoreset(self):
        # self.now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))

        self.image_num = 0
        self.paly_reset_flag = True
        self.pushButton_3.setEnabled(False)

        QApplication.processEvents()

    def detect(self):

        quit_flag = self.softp.is_over_time()
        if quit_flag == True:
            self.timer.start(1000) # 计时1s触发关闭程序
        else:
            if not config.DEBUG:
                from woodev.apis import init_detector, inference_detector
            # build the model from a config file and a checkpoint file
            if self.isimage == True:
                start = time.time()
                if  not config.DEBUG:
                    self.model = init_detector(self.config_file, self.checkpoint_file_list[0])
                    result = inference_detector(self.model, self.org_img)
                else:
                    result = []
                self.time = time.time() - start
                ####得到检测结果
                if not config.DEBUG:
                    result_img, inds, cal_results=new_show_result_2(self.org_img, self.filename,  result, self.class_names, score_thr=0.5)
                    ###将检测结果显示到图窗上
                    result_img = cv2.cvtColor(np.asarray(result_img),cv2.COLOR_RGB2BGR)
                else:
                    result_img = np.zeros((520,520,3),dtype=np.uint8)
                    inds = [0,0]
                    pixels_output = [0,0]

                img = cv2.resize(result_img, (520, 520))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                x = img.shape[1]  # 获取图像大小
                y = img.shape[0]
                self.zoomscale = 1  # 图片放缩尺度
                frame = QImage(img, x, y, QImage.Format_RGB888)
                pix = QPixmap.fromImage(frame)
                self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
                # self.item.setScale(self.zoomscale)
                self.scene = QGraphicsScene()  # 创建场景
                self.scene.addItem(self.item)
                self.graphicsView_2.setScene(self.scene)
                self.textEdit.setText("检测木头个数：" + str(len(inds)) + '\n')
                self.textEdit.append("单张图片检测运行时间:" + str(self.time) + 's' + '\n')
                self.textEdit.append('每根木头的长轴和短轴长度为:\n')
                #print(cal_results)
                [self.textEdit.append('长轴：' + str(cal_result[0] / 10) + 'cm' + ',' + '短轴：' + str(cal_result[1] / 10) + 'cm' + '\n') for cal_result in cal_results]
            elif self.isvideo == True:
                #self.update()
                self.pushButton_2.setText("检测中")
                QApplication.processEvents()
                #self.Video_Detect()

                th = threading.Thread(target=self.Video_Detect())
                th.start()
            elif self.ismulti == True:
                self.multi_image()

    def Video_Detect(self):
        if not config.DEBUG:
            from woodev.apis import init_detector, inference_detector
        self.graphicsView.setScene(None)
        result_dir_path = './results_video/'
        if not os.path.exists(result_dir_path):
            os.mkdir(result_dir_path)

        self.cap = cv2.VideoCapture(self.current_filename)
        video_length = self.cap.get(7)
        #print(video_length)
        frame_num = 0
        start = time.time()
        #print(self.current_filename)

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out_file_name = result_dir_path + self.current_filename.split('/')[-1]
        self.frameRate = self.cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(out_file_name, fourcc, self.frameRate, (520, 480))
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if True == self.stopEvent.is_set():
                 self.stopEvent.clear()
                 self.graphicsView.setScene(None)
                 self.graphicsView_2.setScene(None)
                 QApplication.processEvents()
                 break
            if success:
                #print("processing....")
                frame_num += 1
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                img_i = frame.copy()
                img_i = cv2.resize(img_i, (520, 520))
                img_i = cv2.cvtColor(img_i, cv2.COLOR_BGR2RGB)
                img_i = QImage(img_i, img_i.shape[1], img_i.shape[0], QImage.Format_RGB888)
                pix = QPixmap.fromImage(img_i)
                self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
                # self.item.setScale(self.zoomscale)
                self.scene = QGraphicsScene()  # 创建场景
                self.scene.addItem(self.item)
                self.graphicsView.setScene(self.scene)
                #cv2.waitKey(1)
                if not config.DEBUG:
                    self.model = init_detector(self.config_file, self.checkpoint_file_list[0])
                    result = inference_detector(self.model, frame)
                    result_img, inds, cal_results=new_show_result_21(frame, result, self.class_names, score_thr=config.SCORE_THR)
                else:
                    result = []
                    result_img = frame.copy()
                    inds = [0,0]
                    pixels_output = [0,0]



                ###将检测结果显示到图窗上以及保存
                result_img = cv2.cvtColor(np.array(result_img,dtype=np.uint8),cv2.COLOR_RGB2BGR)
                img_o = cv2.resize(result_img, (520, 520))
                img_o = cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB)
                out.write(img_o)

                img_o = QImage(img_o, img_o.shape[1], img_o.shape[0], QImage.Format_RGB888)
                pix = QPixmap.fromImage(img_o)
                self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
                # self.item.setScale(self.zoomscale)
                self.scene = QGraphicsScene()  # 创建场景
                self.scene.addItem(self.item)
                self.graphicsView_2.setScene(self.scene)
                QApplication.processEvents()
                self.textEdit.setText("当前帧检测木头个数："+ str(len(inds)) + '\n')
                self.textEdit.append('当前帧每根木头的像素的个数:\n')
                [self.textEdit.append('长轴：' + str(cal_result[0] / 10) + 'cm' +  ',' + '短轴：' + str(cal_result[1] / 10) + 'cm' + '\n') for cal_result in cal_results]
                #cv2.waitKey(1)
                #print('hah')
                # while(self.pause_det):
                #
                #     # self.textEdit.setText("检测已经暂停，可继续检测" + '\n')
                #     # # 检测时间为总时长除以视频帧数
                #     # self.time = time.time() - start
                #     # self.textEdit.setText("单帧检测运行时间:" + str(self.time) / frame_num + 's' + '\n')
                #     print('wait')
                    #暂停等待
                if video_length == frame_num:
                    self.stopEvent.clear()
                    self.graphicsView_2.setScene(None)
                    self.graphicsView.setScene(None)
                    self.textEdit.setText("当前视频已经处理完毕，请重新添加视频文件" + '\n')
                    break
            else:
                break
            # 检测时间为总时长除以视频帧数
        self.pushButton_2.setText("检测")
        self.time = time.time() - start
        self.textEdit.setText("单帧检测运行时间:" + str(self.time/ frame_num) + 's' + '\n')
        self.cap.release()
        out.release()
    #打开文件选择框，选择要测试的图片 路径中不能包括中文字符
    def multi_image(self):
        if not config.DEBUG:
            from woodev.apis import init_detector, inference_detector
        self.graphicsView_3.setScene(None)
        QApplication.processEvents()
        img_paths = self.img_list
        img_paths_raw = self.img_list_raw
        out_folder = './results/'
        sticher_folder = './input-42-data/'
        if os.path.exists(sticher_folder):
            shutil.rmtree(sticher_folder)
        if not os.path.exists(sticher_folder):
            os.makedirs(sticher_folder)
        if os.path.exists(out_folder):
            shutil.rmtree(out_folder)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        #shutil.copy('sticher/image-stitching', 'results/image-stitching')
        #shutil.copy('sticher/config.cfg', './config.cfg')
        if len(img_paths_raw) < len(img_paths):
            self.textEdit.setText('深度数据有缺失，请保证每张原图有对应的深度数据，为.tif文件！，如果文件缺失，将默认距离为2m' + '\n')
            QApplication.processEvents()
            img_paths_raw = []
            for i in range(len(img_paths)):
                img_paths_raw.append('none')
        img_count = 0
        mask_index = 0
        sticher_flag = self.lineEdit.text()
        for img_path, img_path_raw in zip(img_paths, img_paths_raw):

            out_file = out_folder + img_path.split('/')[-1]
            img = cv2.imread(img_path)
            if not config.DEBUG:
                self.model = init_detector(self.config_file, self.checkpoint_file_list[0])
                result = inference_detector(self.model, img)
            else:
                result = []
            ####得到检测结果
            # if(img_count%2==0):
            #     mask_index=0
            # else:
            #     mask_index=1
            if sticher_flag is "":
                self.textEdit.setText('拼接标志为为空，请输入正确的值！')
                QApplication.processEvents()
            else:
                if int(sticher_flag) <= 0 or str(int(sticher_flag)) != sticher_flag:
                    self.textEdit.setText('拼接顺序值有误，必须为大于0的整数，请重新输入！')
                    QApplication.processEvents()
                else:
                    if  int(sticher_flag)== 1:
                        mask_index = img_count % 3
                        img_count += 1
                    elif int(sticher_flag) == 2:
                        #one_row_num = len(img_paths) // 2
                        mask_index = img_count % 6
                        mask_index = mask_index // 2
                        img_count += 1
                    if not config.DEBUG:
                        _ = new_show_result_3(img, result, self.class_names, img_path_raw,
                                                                          score_thr=0.3, out_file=out_file, mask_index=mask_index)
                        ###将检测结果显示到图窗上
                    else:
                        result_img = np.zeros((520, 520, 3), dtype=np.uint8)
                        inds = [0, 0]
                        pixels_output = [0, 0]
        #print(img_paths)

        # print('sticher_flag:', sticher_flag)
        # print(sticher_flag is None)
        # print(int(sticher_flag) <= 0)
        # print(int(sticher_flag) != sticher_flag)
        if sticher_flag is "":
            self.textEdit.setText('拼接标志为为空，请输入正确的值！')
            QApplication.processEvents()
        else:
            if int(sticher_flag) <= 0 or str(int(sticher_flag)) != sticher_flag:
                self.textEdit.setText('拼接顺序值有误，必须为大于0的整数，请重新输入！')
                QApplication.processEvents()
            else:
                if int(sticher_flag) == 1:
                    commad = "./sticher/NISwGSP"
                    file_name = img_paths[0].split('/')[-3] + '-' + img_paths[0].split('/')[-2]
                    sub_siticher_folders = sticher_folder + file_name
                    sub_siticher_folders_mask = sub_siticher_folders + '-mask'
                    if not os.path.exists(sub_siticher_folders):
                        os.makedirs(sub_siticher_folders)
                    if not os.path.exists(sub_siticher_folders_mask):
                        os.makedirs(sub_siticher_folders_mask)
                    print(img_paths)
                    for img_path in img_paths:
                        filename = img_path.split('/')[-1]
                        shutil.copy(out_folder + filename, sub_siticher_folders + '/' + filename)
                        filename_mask = filename.split('.')[0] + '_mask.png'
                        print(filename_mask)
                        shutil.copy(out_folder + filename_mask, sub_siticher_folders_mask + '/' + filename_mask)
                    #print(commad)
                    file_name_txt = file_name +'-STITCH-GRAPH'
                    creat_match_graph_txt(img_count, root_path=sub_siticher_folders, root_path_mask=sub_siticher_folders_mask, file_name=file_name_txt)
                    commad += ' ' + file_name
                    cmd = os.system(commad)
                    if cmd != 0:
                        self.textEdit.setText('图片数据有误！' + '\n')
                        QApplication.processEvents()

                    ###########################################拼图模块
                    #保存拼接的txt对应信息

                    #调用拼接程序 进行拼接
                    #需要将拼接后的图片和掩码移动到 results 文件夹中
                    #
                    shutil.move(sticher_folder + '0_results/' + file_name  + '-result' + '/'  + file_name + '-[NISwGSP][2D][BLEND_LINEAR].png', out_folder + 'out.png')
                    shutil.move(sticher_folder + '0_results/' + file_name +  '-mask-result' + '/'  + file_name + '-[NISwGSP][2D][BLEND_LINEAR].png', out_folder + 'out_mask.png')
                    #print('Done!')


                    #############################################结果整合模块,，函数定义在merger_result.py
                    #得到保存的结果文件列表  检测csv文件列表 拼接的图片名称

                    input_image_list =  [out_folder + img_path.split('/')[-1] for img_path in img_paths]
                    input_csv_file_list = [ img_path[:-4]+'.csv' for img_path in input_image_list]
                    out_final_mask = out_folder + 'out_mask.png'


                    log_info = get_final_result(input_image_list,input_csv_file_list, out_final_mask)#这个函数运行结束后会保存merger_final.csv作为最终的结果
                    print('log_info:', log_info)
                    if log_info == '正常':
                        img, cal_list, count_num = get_result_from_meger_file('./meger_final.csv', out_folder +'out.png')
                        img1 = cv2.imread(out_folder + 'out.png')
                        img2 = img
                        img = cv2.resize(img2, (1100, 550))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        frame = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
                        pix = QPixmap.fromImage(frame)
                        self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
                        # self.item.setScale(self.zoomscale)
                        self.scene = QGraphicsScene()  # 创建场景
                        self.scene.addItem(self.item)
                        self.graphicsView_3.setScene(self.scene)
                        # img1 = cv2.resize(img1, (520, 520))
                        # img2 = cv2.resize(img2, (520, 520))
                        # x = img1.shape[1]  # 获取图像大小
                        # y = img1.shape[0]
                        # self.zoomscale = 1  # 图片放缩尺度
                        # frame1 = QImage(img1, x, y, QImage.Format_RGB888)
                        # pix1 = QPixmap.fromImage(frame1)
                        # self.item1 = QGraphicsPixmapItem(pix1)  # 创建像素图元
                        # # self.item.setScale(self.zoomscale)
                        # self.scene1 = QGraphicsScene()  # 创建场景
                        # self.scene1.addItem(self.item1)
                        #
                        # frame2 = QImage(img2, x, y, QImage.Format_RGB888)
                        # pix2 = QPixmap.fromImage(frame2)
                        # self.item2 = QGraphicsPixmapItem(pix2)  # 创建像素图元
                        # # self.item.setScale(self.zoomscale)
                        # self.scene2 = QGraphicsScene()  # 创建场景
                        # self.scene2.addItem(self.item2)
                        #
                        # self.graphicsView.setScene(self.scene1)
                        # self.graphicsView_2.setScene(self.scene2)


                        self.textEdit.setText("检测木头总的个数：" + str(count_num) + '\n')
                        self.textEdit.append('具体的结果保存在results文件夹,以及meger_final.csv中:\n')
                        [self.textEdit.append('第' + str(i)  + '根木头:\n' + '长轴：' + str(cal_result[0] / 10) + 'cm' + ',' + '\n短轴：' + str(cal_result[1] / 10)  + 'cm' + '\n') for i, cal_result in enumerate(cal_list)]
                        QApplication.processEvents()
                        #这里读取csv的到木头的个数 该文件的注释格式和之前检测基本相同  最后多了四个项 为 该木头在拼接后的图中的bbox
                    else:
                        self.textEdit.setText(str(log_info) + '\n')
                        QApplication.processEvents()

                elif  int(sticher_flag) == 2:
                    file_name = img_paths[0].split('/')[-3] + '-' + img_paths[0].split('/')[-2]
                    img_paths_odd = img_paths[::2]
                    img_paths_even = img_paths[1::2]
                    img_paths_raw_odd = img_paths_raw[::2]
                    img_paths_raw_even = img_paths_raw[1::2]
                    commad = "./sticher/NISwGSP"

                    sub_siticher_folders = sticher_folder + file_name
                    sub_siticher_folders_mask = sub_siticher_folders + '-mask'

                    sub_siticher_folders1 = sticher_folder + file_name+'_1'
                    sub_siticher_folders_mask1 = sub_siticher_folders1 + '-mask'

                    if not os.path.exists(sub_siticher_folders):
                        os.makedirs(sub_siticher_folders)
                    if not os.path.exists(sub_siticher_folders1):
                        os.makedirs(sub_siticher_folders1)
                    if not os.path.exists(sub_siticher_folders_mask):
                        os.makedirs(sub_siticher_folders_mask)
                    if not os.path.exists(sub_siticher_folders_mask1):
                        os.makedirs(sub_siticher_folders_mask1)

                    count=0
                    for img_path in img_paths:
                        filename = img_path.split('/')[-1]
                        filename_mask = filename.split('.')[0] + '_mask.png'
                        filename_bmask = filename.split('.')[0] + '_bmask.png'

                        shutil.copy(out_folder + filename, sub_siticher_folders + '/' + filename)
                        shutil.copy(out_folder + filename, sub_siticher_folders1 + '/' + filename)

                        if count%2==0:
                            shutil.copy(out_folder + filename_mask, sub_siticher_folders_mask + '/' + filename_mask)
                            shutil.copy(out_folder + filename_bmask, sub_siticher_folders_mask1 + '/' + filename_mask)
                        else:
                            shutil.copy(out_folder + filename_mask, sub_siticher_folders_mask1 + '/' + filename_mask)
                            shutil.copy(out_folder + filename_bmask, sub_siticher_folders_mask + '/' + filename_mask)
                        count+=1
                    file_name_txt = file_name + '-STITCH-GRAPH'
                    creat_match_graph_txt(img_count, root_path=sub_siticher_folders,
                                          root_path_mask=sub_siticher_folders_mask, file_name=file_name_txt)
                    file_name_1 =  file_name + '_1'
                    file_name_txt = file_name + '_1' + '-STITCH-GRAPH'
                    creat_match_graph_txt(img_count, root_path=sub_siticher_folders1,
                                          root_path_mask=sub_siticher_folders_mask1, file_name=file_name_txt)
                    commad_1 = commad + ' ' + file_name
                    cmd = os.system(commad_1)
                    if cmd != 0:
                        self.textEdit.setText('图片数据有误！' + '\n')
                        QApplication.processEvents()
                    commad_1 = commad + ' ' + file_name+'_1'
                    cmd = os.system(commad_1)
                    if cmd != 0:
                        self.textEdit.setText('图片数据有误！' + '\n')
                        QApplication.processEvents()

                    shutil.copy(
                        sticher_folder + '0_results/' + file_name + '-result' + '/' + file_name + '-[NISwGSP][2D][BLEND_LINEAR].png',
                        out_folder + 'out1.png')
                    shutil.copy(
                        sticher_folder + '0_results/' + file_name + '-mask-result' + '/' + file_name + '-[NISwGSP][2D][BLEND_LINEAR].png',
                        out_folder + 'out_mask1.png')
                    shutil.copy(
                        sticher_folder + '0_results/' + file_name_1 + '-result' + '/' + file_name_1 + '-[NISwGSP][2D][BLEND_LINEAR].png',
                        out_folder + 'out2.png')
                    shutil.copy(
                        sticher_folder + '0_results/' + file_name_1 + '-mask-result' + '/' + file_name_1 + '-[NISwGSP][2D][BLEND_LINEAR].png',
                        out_folder + 'out_mask2.png')
                    # input_image_list = ['./tmp_merger_final1.png', './tmp_merger_final2.png']
                    # input_csv_file_list = ['./tmp_merger_final1.csv', './tmp_merger_final2.csv']
                    # out_final_mask =  out_folder + 'out_mask.png'
                    out_final_mask = [out_folder + 'out_mask1.png',
                                      out_folder + 'out_mask2.png']
                    input_csv_file_list = [out_folder + img_path.split('/')[-1].rstrip('.png') + '.csv' for img_path in img_paths]
                    input_img_mask_list = [out_folder + img_path.split('/')[-1] for img_path in img_paths]
                    log_info = get_final_result_multirows(input_img_mask_list, input_csv_file_list, out_final_mask)
                    if log_info == '正常':
                        img, cal_list, count_num = get_result_from_meger_file('./' +'meger_final.csv', out_folder + 'out2.png')
                        img1 = cv2.imread(out_folder + 'out2.png')
                        img2 = img
                        x = img1.shape[1]  # 获取图像大小
                        y = img1.shape[0]
                        img = cv2.resize(img2, (1100, 550))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        frame = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
                        pix = QPixmap.fromImage(frame)
                        self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
                        # self.item.setScale(self.zoomscale)
                        self.scene = QGraphicsScene()  # 创建场景
                        self.scene.addItem(self.item)
                        self.graphicsView_3.setScene(self.scene)
                        # self.zoomscale = 1  # 图片放缩尺度
                        # frame1 = QImage(img1, x, y, QImage.Format_RGB888)
                        # pix1 = QPixmap.fromImage(frame1)
                        # self.item1 = QGraphicsPixmapItem(pix1)  # 创建像素图元
                        # # self.item.setScale(self.zoomscale)
                        # self.scene1 = QGraphicsScene()  # 创建场景
                        # self.scene1.addItem(self.item1)
                        #
                        # frame2 = QImage(img2, x, y, QImage.Format_RGB888)
                        # pix2 = QPixmap.fromImage(frame2)
                        # self.item2 = QGraphicsPixmapItem(pix2)  # 创建像素图元
                        # # self.item.setScale(self.zoomscale)
                        # self.scene2 = QGraphicsScene()  # 创建场景
                        # self.scene2.addItem(self.item2)
                        #
                        # self.graphicsView.setScene(self.scene1)
                        # self.graphicsView_2.setScene(self.scene2)
                        self.textEdit.setText("检测木头总的个数：" + str(count_num) + '\n')
                        self.textEdit.append('具体的结果保存在results文件夹,以及meger_final.csv中:\n')
                        [self.textEdit.append(
                            '第' + str(i) + '根木头:\n' + '长轴：' + str(cal_result[0] / 10) + 'cm' +  ',' + '\n短轴：' + str(cal_result[1] / 10) + 'cm' + '\n') for
                         i, cal_result in enumerate(cal_list)]
                        QApplication.processEvents()
                    else:
                        self.textEdit.setText(str(log_info) + '\n')
                        QApplication.processEvents()


                    ####################################################################old version
                    # sub_siticher_folders = sticher_folder + file_name
                    # sub_siticher_folders_mask = sub_siticher_folders + '-mask'
                    # if not os.path.exists(sub_siticher_folders):
                    #     os.makedirs(sub_siticher_folders)
                    # if not os.path.exists(sub_siticher_folders_mask):
                    #     os.makedirs(sub_siticher_folders_mask)
                    # # print("img_paths:", img_paths)
                    # # print('odd:', img_paths_odd)
                    # # print("even:", img_paths_even)
                    # for img_path_odd in img_paths_odd:
                    #     filename = img_path_odd.split('/')[-1]
                    #     shutil.copy(out_folder + filename, sub_siticher_folders + '/' + filename)
                    #     filename_mask = filename.split('.')[0] + '_mask.png'
                    #     # print(filename_mask)
                    #     shutil.copy(out_folder + filename_mask, sub_siticher_folders_mask + '/' + filename_mask)
                    # # print(commad)
                    # file_name_txt = file_name + '-STITCH-GRAPH'
                    # creat_match_graph_txt(img_count // 2, root_path=sub_siticher_folders,
                    #                       root_path_mask=sub_siticher_folders_mask, file_name=file_name_txt)
                    # commad_1 = commad + ' ' + file_name
                    # cmd = os.system(commad_1)
                    # if cmd != 0:
                    #     self.textEdit.setText('图片数据有误！' + '\n')
                    #     QApplication.processEvents()
                    #
                    # # second image ###
                    # file_name_2 = file_name +'_2'
                    # sub_siticher_folders_2 = sticher_folder + file_name_2
                    # sub_siticher_folders_mask_2 = sub_siticher_folders_2 + '-mask'
                    # if not os.path.exists(sub_siticher_folders_2):
                    #     os.makedirs(sub_siticher_folders_2)
                    # if not os.path.exists(sub_siticher_folders_mask_2):
                    #     os.makedirs(sub_siticher_folders_mask_2)
                    # print(img_paths)
                    # for img_path_even in img_paths_even:
                    #     filename = img_path_even.split('/')[-1]
                    #     shutil.copy(out_folder + filename, sub_siticher_folders_2 + '/' + filename)
                    #     filename_mask = filename.split('.')[0] + '_mask.png'
                    #     # print(filename_mask)
                    #     shutil.copy(out_folder + filename_mask, sub_siticher_folders_mask_2 + '/' + filename_mask)
                    # # print(commad)
                    # file_name_txt = file_name_2 + '-STITCH-GRAPH'
                    # creat_match_graph_txt(img_count // 2, root_path=sub_siticher_folders_2,
                    #                       root_path_mask=sub_siticher_folders_mask_2, file_name=file_name_txt)
                    #
                    # commad_2 =  commad + ' ' + file_name_2
                    # cmd = os.system(commad_2)
                    # if cmd != 0:
                    #     self.textEdit.setText('图片数据有误！' + '\n')
                    #     QApplication.processEvents()
                    #
                    # file_name_3 = file_name + '_3'
                    # sub_siticher_folders_3 = sticher_folder + file_name_3
                    # sub_siticher_folders_mask_3 = sub_siticher_folders_3 + '-mask'
                    # if not os.path.exists(sub_siticher_folders_3):
                    #     os.makedirs(sub_siticher_folders_3)
                    # if not os.path.exists(sub_siticher_folders_mask_3):
                    #     os.makedirs(sub_siticher_folders_mask_3)
                    # shutil.move(
                    #     sticher_folder + '0_results/' + file_name + '-result' + '/' + file_name + '-[NISwGSP][2D][BLEND_LINEAR].png',
                    #     sub_siticher_folders_3 + '/' + 'out1.png')
                    #
                    # shutil.move(
                    #     sticher_folder + '0_results/' + file_name_2 + '-result' + '/' + file_name_2 + '-[NISwGSP][2D][BLEND_LINEAR].png',
                    #     sub_siticher_folders_3 + '/' +  'out2.png')
                    #
                    # out_final_mask = [sticher_folder + '0_results/' + file_name + '-mask-result' + '/' + file_name + '-[NISwGSP][2D][BLEND_LINEAR].png',
                    #                   sticher_folder + '0_results/' + file_name_2 + '-mask-result' + '/' + file_name_2 + '-[NISwGSP][2D][BLEND_LINEAR].png']
                    # input_csv_file_list = [out_folder + img_path.split('/')[-1].rstrip('.png') + '.csv' for img_path in img_paths]
                    # input_img_mask_list = [out_folder + img_path.split('/')[-1] for img_path in img_paths]
                    # get_final_result_multirows(input_img_mask_list, input_csv_file_list, out_final_mask)
                    # shutil.copy(os.getcwd() + '/' + 'tmp_merger_final1.png', sub_siticher_folders_mask_3 + '/' + 'out_mask1.png')
                    # shutil.copy(os.getcwd() + '/' + 'tmp_merger_final2.png', sub_siticher_folders_mask_3 + '/' + 'out_mask2.png')
                    # file_name_txt = file_name_3 + '-STITCH-GRAPH'
                    # creat_match_graph_txt(2, root_path=sub_siticher_folders_3,
                    #                       root_path_mask=sub_siticher_folders_mask_3, file_name=file_name_txt)
                    # commad_3 = commad + ' ' + file_name_3
                    # cmd = os.system(commad_3)
                    # if cmd != 0:
                    #     self.textEdit.setText('图片数据有误！' + '\n')
                    #     QApplication.processEvents()
                    # shutil.move(
                    #     sticher_folder + '0_results/' + file_name_3 + '-result' + '/' + file_name_3 + '-[NISwGSP][2D][BLEND_LINEAR].png',
                    #     out_folder + 'out.png')
                    # shutil.move(
                    #     sticher_folder + '0_results/' + file_name_3 + '-mask-result' + '/' + file_name_3 + '-[NISwGSP][2D][BLEND_LINEAR].png',
                    #     out_folder + 'out_mask.png')
                    # input_image_list = ['./tmp_merger_final1.png', './tmp_merger_final2.png']
                    # input_csv_file_list = ['./tmp_merger_final1.csv', './tmp_merger_final2.csv']
                    # out_final_mask =  out_folder + 'out_mask.png'
                    # log_info = get_final_result(input_image_list, input_csv_file_list, out_final_mask, is_row_format=False)
                    # if log_info == '正常':
                    #     img, cal_list, count_num = get_result_from_meger_file('./meger_final.csv', out_folder + 'out.png')
                    #     img1 = cv2.imread(out_folder + 'out.png')
                    #     img2 = img
                    #     x = img1.shape[1]  # 获取图像大小
                    #     y = img1.shape[0]
                    #     img = cv2.resize(img2, (1100, 550))
                    #     frame = QImage(img, img.shape[1], img.shape[0], QImage.Format_RGB888)
                    #     pix = QPixmap.fromImage(frame)
                    #     self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
                    #     # self.item.setScale(self.zoomscale)
                    #     self.scene = QGraphicsScene()  # 创建场景
                    #     self.scene.addItem(self.item)
                    #     self.graphicsView_3.setScene(self.scene)
                    #     # self.zoomscale = 1  # 图片放缩尺度
                    #     # frame1 = QImage(img1, x, y, QImage.Format_RGB888)
                    #     # pix1 = QPixmap.fromImage(frame1)
                    #     # self.item1 = QGraphicsPixmapItem(pix1)  # 创建像素图元
                    #     # # self.item.setScale(self.zoomscale)
                    #     # self.scene1 = QGraphicsScene()  # 创建场景
                    #     # self.scene1.addItem(self.item1)
                    #     #
                    #     # frame2 = QImage(img2, x, y, QImage.Format_RGB888)
                    #     # pix2 = QPixmap.fromImage(frame2)
                    #     # self.item2 = QGraphicsPixmapItem(pix2)  # 创建像素图元
                    #     # # self.item.setScale(self.zoomscale)
                    #     # self.scene2 = QGraphicsScene()  # 创建场景
                    #     # self.scene2.addItem(self.item2)
                    #     #
                    #     # self.graphicsView.setScene(self.scene1)
                    #     # self.graphicsView_2.setScene(self.scene2)
                    #     self.textEdit.setText("检测木头总的个数：" + str(count_num) + '\n')
                    #     self.textEdit.append('具体的结果保存在results文件夹,以及meger_final.csv中:\n')
                    #     [self.textEdit.append(
                    #         '第' + str(i) + '根木头:\n' + '长轴：' + str(cal_result[0] / 10) + 'cm' +  ',' + '\n短轴：' + str(cal_result[1] / 10) + 'cm' + '\n') for
                    #      i, cal_result in enumerate(cal_list)]
                    #     QApplication.processEvents()
                    # else:
                    #     self.textEdit.setText(str(log_info) + '\n')
                    #     QApplication.processEvents()

                else:
                    self.textEdit.setText("拼接顺序仅支持一/二排，请重新输入！！")
                    QApplication.processEvents()




    def open_file_and_show_img(self):
        self.graphicsView_2.setScene(None)
        if self.isimage == True:
            self.textEdit.setText("开始检测图片，选择要检测的图片，选择前请勾选对应选项如（图片/视频）, 且保证存在.tif的深度数据")
            file = QFileDialog.getOpenFileName(self, "Open File", "./","Images (*.png *.xpm *.jpg)")
            current_filename = file[0]
            current_filename = str(current_filename)
            self.filename = current_filename
            img = cv2.imread(current_filename)
            if img is None:
                print('读取图片为空!!!')
                self.graphicsView_2.setScene(None)
                self.graphicsView.setScene(None)
                self.textEdit.setText("输入图片为空，请重新输入" + '\n')
            else:
                self.org_img = img.copy() #得到原始图片
                img = cv2.resize(img,(520,520))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                x = img.shape[1]  # 获取图像大小
                y = img.shape[0]
                self.zoomscale = 1  # 图片放缩尺度

                frame = QImage(img, x, y, QImage.Format_RGB888)
                pix = QPixmap.fromImage(frame)
                self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
                # self.item.setScale(self.zoomscale)
                self.scene = QGraphicsScene()  # 创建场景
                self.scene.addItem(self.item)
                self.graphicsView.setScene(self.scene)
    
        elif self.isvideo == True:
            self.textEdit.setText("开始检测视频，选择要检测的视频，选择前请勾选对应选项（图片/视频），仅支持mp4视频")
            file = QFileDialog.getOpenFileName(self, "Open File", "./","*.mp4")
            current_filename = file[0]
            #保存文件路径
            self.current_filename = str(current_filename)
            self.cap = cv2.VideoCapture(self.current_filename )
            self.frameRate = self.cap.get(cv2.CAP_PROP_FPS)
            while(self.cap.isOpened()):
                # 如果读取成功
                success, frame = self.cap.read()
                if(success):
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_i = frame.copy()
                    img_i = cv2.resize(img_i, (520, 520))
                    img_i = cv2.cvtColor(img_i, cv2.COLOR_BGR2RGB)
                    new_frame = QImage(img_i, img_i.shape[1],img_i.shape[0] , QImage.Format_RGB888)
                    pix = QPixmap.fromImage(new_frame)
                    self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
                    # self.item.setScale(self.zoomscale)
                    self.scene = QGraphicsScene()  # 创建场景
                    self.scene.addItem(self.item)
                    self.graphicsView.setScene(self.scene)
                    cv2.waitKey(1)
                    break
                else:
                    break
            self.cap.release()
        elif self.ismulti == True:
            self.textEdit.setText("选择要检测的所有帧的文件夹，开始多帧融合并检测，请确保每帧是按顺序编号，选择前请勾选对应选项（图片/视频/多帧图像）,保证tif深度数据存在同一个文件夹中")
            file = QFileDialog.getExistingDirectory(self,
                  "选取文件夹",
                  "./")                 #起始路径
            #print(file)
            directory = file
            if directory is '':
                self.textEdit.setText("选择文件夹有误，请重新选择" + '\n')
                QApplication.processEvents()
            else:
                #print('directory:', directory)
                filenames = os.listdir(directory)
                filenames_new = []
                filenames_raw = []
                is_sucess = True
                for filename in filenames:
                    if 'tif' in filename or 'png' in filename:
                        if 'tif' in filename:
                            filenames_raw.append(filename)
                            continue
                        if 'depth' in filename:
                            continue
                        filenames_new.append(filename)
                    else:
                        is_sucess = False
                if len(filenames_new) != len(filenames_raw):
                    is_sucess = False
                if is_sucess == True:
                    filenames_new = sorted(filenames_new, key = lambda x: int(x[:-4]))
                    print('filename:', filenames_new)
                    filenames_raw = sorted(filenames_raw, key = lambda x: int(x[:-10]))
                    #print('filenames:', filenames)
                    img_list = [directory + '/' + filename for filename in filenames_new]
                    img_list_raw = [directory + '/' + filename for filename in filenames_raw]
                    self.img_list = img_list
                    self.img_list_raw = img_list_raw
                else:
                    self.textEdit.setText("选择文件夹有误，请重新选择" + '\n')
                    QApplication.processEvents()

    #在指定图窗上显示特定的图像
    def show_img_in_graphics(self,img,graphicsView):
        pass



# if  __name__=="__main__" and config.DEBUG:
#     import  sys
#     print("hello world")
#     # cap = cv2.VideoCapture('./1.mp4')
#     # frameRate = cap.get(cv2.CAP_PROP_FPS)
#     # while cap.isOpened():
#     #     success, frame = cap.read()
#     #     if(success):
#     #         print(frame.shape)
#
#     app = QApplication(sys.argv)
#     w = Detection_Ui()
#
#     w.show()
#     #if w.ismulti == True:
#     sys.exit(app.exec_())

