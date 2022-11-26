#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
这个代码用来耦合音视频录入，创造两线程来同步进行
This function is to call the mediapipe and to record the real time audio simutaneously And return the coordinates of the face mesh and audio MFCC features.
"""

import mediapipe as mp
import time
import cv2
import numpy as np
import pyaudio
import wave
import time
import librosa as lr
import matplotlib.pyplot as plt
import librosa.display
import threading
import os
from spafe.features.gfcc import gfcc
from scipy.signal import hilbert
from scipy.signal import butter

class VideoRecorder(threading.Thread):
    def __init__(self,camindex=0):
        self.open = True
        self.device_index = camindex
        self.video_cap = cv2.VideoCapture(self.device_index)
        self.result = None

    def record(self):  # 用mediapipe库获取人脸网格数据
        # 调用mediapipe库里的脸部检测，并定义一个face人脸检测模型
        global X_v
        mp_face_detection = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils

        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh
        # %%

        # For webcam input:
        # 使用draingspec来定义线的粗细，颜色信息等
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        pTime = 0
        cap = cv2.VideoCapture(0)  # 获取视频对象，0为摄像头
        begin = time.time()
        result = []
        # 使用mp_face_mesh.FaceMesh方法定义一个人脸检测器与人脸mesh器
        with mp_face_mesh.FaceMesh(
                max_num_faces=1,  # 最大检测脸数
                # 是否更进一步细化眼睛和嘴唇周围的地标坐标，并在虹膜周围输出额外的地标
                # 我们这里需要用到眼睛和嘴唇周围的坐标，所以设置为True
                refine_landmarks=True,
                min_detection_confidence=0.5,  # 检测模型的最小置信值，为检测成功
                min_tracking_confidence=0.5) as face_mesh:  # 地标跟踪模型的最小置信值，越高越稳健但是有更高的延迟

            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # 这里用process方法对图片进行检测，返回人脸468个点的坐标，并遍历所有点，进行人脸mesh的绘制
                results = face_mesh.process(image)

                # Draw the face mesh annotations on the image.
                # 对摄像机获得的输入视频流做一些预处理
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # BGR图转RGB图
                faces = []
                if results.multi_face_landmarks:  # 如果识别出人脸
                    # 遍历所有点，绘制
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=image,  # 需要画图的原始图片
                            landmark_list=face_landmarks,  # 检测到的人脸坐标
                            connections=mp_face_mesh.FACEMESH_TESSELATION,  # 绘制人脸网络
                            landmark_drawing_spec=None,  # 为关键点可视化样式，None为默认
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_tesselation_style())
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,  # 绘制人脸轮廓，眼睫毛，眼眶，嘴唇
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_contours_style())
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_IRISES,  # 绘制瞳孔区域
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_iris_connections_style())
                        face = []
                        for id, lm in enumerate(face_landmarks.landmark):
                            # print(lm)
                            ih, iw, ic = image.shape
                            x, y = int(lm.x * iw), int(lm.y * ih)
                            # print(id, lm.x, lm.y, lm.z)
                            face.append([lm.x, lm.y, lm.z])
                            # print(face)
                        faces.append(face)
                        result.append(faces)
                # Flip the image horizontally for a selfie-view display.
                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime
                cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                            3, (0, 255, 0), 3)
                cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
                finish = time.time()
                duration = int(finish) - int(begin)
                #if duration >= 5:
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
        cap.release()
        # cv2.destroyAllWindows()

        faces = np.array(faces)
        X_v = np.array(result)
        X_v = X_v.squeeze()

        # print(faces)
        # print(np.size(faces))
        # print(result)
        # print(np.size(result))
        FPS = np.size(X_v) / np.size(faces) / duration
        print('The average FPS of %ds recording is %d' % (duration, FPS))
        # 到这里我们已经得到了录制5s的人脸网格数据，并存储在X_v中
        # return X_v
        self.result = X_v
        # time.sleep(1)


    def start(self):
        "Launches the vedio recording function using a thread"
        video_thread = threading.Thread(target=self.record())
        video_thread.start()

    def stop(self):
        "Finishes the video recording therefore the thread too"
        if self.open:
            self.open=False
            self.video_cap.release()
            cv2.destroyAllWindows()

    def return_results(self):
        return self.result


class AudioRecorder():
    "Audio class based on pyAudio and Wave"
    def __init__(self, filename="temp_audio.wav", rate=44100, fpb=1024, channels=1):
        self.open = True
        self.rate = rate
        self.frames_per_buffer = fpb
        self.channels = channels
        self.format = pyaudio.paInt16
        self.audio_filename = filename
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer = self.frames_per_buffer)
        self.audio_frames = []
        self.result = None
    def record(self):
        "Audio starts being recorded"
        self.stream.start_stream()
        while self.open:
            data = self.stream.read(self.frames_per_buffer)
            self.audio_frames.append(data)
            if not self.open:
                break
        self.result = self.audio_frames
    def stop(self):
        "Finishes the audio recording therefore the thread too"
        if self.open:
            self.open = False
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
            waveFile = wave.open(self.audio_filename, 'wb')
            waveFile.setnchannels(self.channels)
            waveFile.setsampwidth(self.audio.get_sample_size(self.format))
            waveFile.setframerate(self.rate)
            waveFile.writeframes(b''.join(self.audio_frames))
            waveFile.close()

    def start(self):
        "Launches the audio recording function using a thread"
        audio_thread = threading.Thread(target=self.record)
        audio_thread.start()

    def return_results(self):
        return self.result


"Using for regularizing the data"
def LagGenerator(arr, lags):
    nlag = len(lags)
    n = arr.shape
    if len(n) > 1:
        n, dim = arr.shape
    else:
        dim = 1
        n = len(arr)
    out = np.zeros((n, dim * nlag))
    idx = 0
    for count, lag in enumerate(lags):
        t1 = np.roll(arr, lag, axis=0)
        if dim == 1:
            t1 = t1[:, None]
        if lag < 0:
            t1[-abs(lag):, :] = 0
        else:
            t1[:lag, :] = 0
        out[:, idx:idx + dim] = t1[:n, :]
        idx = idx + dim
    return out

#
# def start_audio_recording(filename="test"):
#     global audio_thread
#     audio_thread = AudioRecorder()
#     audio_thread.start()
#
#     return filename
#
# def start_video_recording():
#     global vedio_thread
#     vedio_thread = VideoRecorder()
#     video_thread.start()

def start_AVrecording(filename="test"):
    global audio_thread
    global video_thread
    audio_thread = AudioRecorder()
    video_thread = VideoRecorder()
    print('Attention please:Recording will start in 3s')
    time_left = 3
    for i in range(0, time_left):
        print(time_left)
        time.sleep(1)
        time_left = time_left - 1
    print('Start recording')
    audio_thread.start()
    video_thread.start()
    au_result = audio_thread.return_results()
    vi_result = video_thread.return_results()
    # print(vi_result.shape)
    # print(vi_result)
    return vi_result

def stop_AVrecording(filename="test"):
    audio_thread.stop()
    video_thread.stop()
    print('Recording end')
    return filename

def get_Audiofeatures(filename,rate):
    snd = lr.load(filename, sr=rate, mono=True)[0]
    "plot the singal figure"
    sr = rate
    duration = 5
    samples = int(rate*duration)
    t = np.arange(samples)/rate
    length = len(snd)
    t = t[:length]
    # plt.figure()
    analytic_signal = hilbert(snd)
    amplitude_envelope = np.abs(analytic_signal)
    # plt.plot(t,snd,label='signal')
    # plt.plot(t, amplitude_envelope, label='envelope')
    plt.figure()
    lr.display.waveshow(snd, sr)
    plt.title('Amplitude')
    "plot the STFT"
    plt.figure()
    data = lr.amplitude_to_db(np.abs(lr.stft(snd,n_fft =512, hop_length=None, win_length=None,window='hann')),ref = np.max)
    img = lr.display.specshow(data,sr= sr,x_axis = 'time',y_axis = 'linear')
    plt.title('STFT')
    plt.colorbar()
    "plot the MFCC"
    mf = lr.feature.mfcc(y = snd,sr = rate, n_mfcc = 14,hop_length=641,htk=False).T
    mf = LagGenerator(mf, np.array([-1]))
    logmf = lr.power_to_db(mf)
    plt.figure()
    lr.display.specshow(logmf, sr=sr, x_axis = 'time',y_axis ='mel')
    plt.title('Mel spectrom')
    plt.show()
    "Get gfccs"
    gfccs = gfcc(snd)
    envelope_gfccs = hilbert(gfccs)
    return envelope_gfccs

def load_features(Audio_data,Vedio_data):
    "Create an empty data set"
    Xv = []
    Xa = []
    Xv = np.linalg.norm(Vedio_data-Vedio_data[:,7,:][:,np.newaxis,:],axis=2)
    if len(Audio_data) == len(Vedio_data):
        Xa.append(Audio_data)
        Xv.append((Video_data))
    return Xv,Xa



if __name__ == '__main__':
    vi_result = start_AVrecording()
    # time.sleep(5)
    stop_AVrecording()
    filename = "temp_audio.wav"
    X_A = get_Audiofeatures(filename ,16000)
    X_V = vi_result
    # print(X_V.shape)
    "Now, the audio and video signal has been captured"
    "Then consider get features from both signals"
    # Xv, Xa = load_features(X_A,X_V)
    #
    # print(Xv)
    print(len(X_A))
    print(len(X_V))
    time.sleep(1)







