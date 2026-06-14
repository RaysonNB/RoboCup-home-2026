#!/usr/bin/env python3.8
import rospy
import os
import time
import numpy as np
import librosa
from std_msgs.msg import String
from mr_voice.msg import Voice

TARGET_PATH = "/home/pcms/robot/doorbell/dragon-studio-doorbell-ding-dong-482879 (2).mp3"
SAMPLE_RATE = 44100
LOG_PATH = "/tmp/bell_debug_log.txt"

target_mfcc = None
last_audio_path = None


def get_mfcc(y):
    return librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=40).T


def mfcc_distance(audio_path):
    global target_mfcc

    audio_y, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    audio_mfcc = get_mfcc(audio_y)

    n = min(len(audio_mfcc), len(target_mfcc))
    if n <= 0:
        return None

    a = audio_mfcc[-n:]
    t = target_mfcc[-n:]

    distance = np.sum(np.sqrt((a[:, 1:] - t[:, 1:]) ** 2))
    return float(distance)


def callback_path(msg):
    global last_audio_path

    audio_path = msg.data

    if audio_path == last_audio_path:
        return

    last_audio_path = audio_path

    print("\n========== NEW AUDIO ==========")
    print("path =", audio_path)

    if not os.path.exists(audio_path):
        print("exists = NO")
        return

    size = os.path.getsize(audio_path)
    print("size =", size)

    if size < 1000:
        print("file too small")
        return

    try:
        distance = mfcc_distance(audio_path)
        print("distance =", distance)

        with open(LOG_PATH, "a") as f:
            f.write(f"{time.time()},{audio_path},{size},{distance}\n")

        print("logged to", LOG_PATH)

    except Exception as e:
        print("MFCC error:", e)


def callback_voice(msg):
    print("\n========== VOICE TEXT ==========")
    print("text =", msg.text)


if __name__ == "__main__":
    rospy.init_node("bell_debug")

    print("Loading target bell...")
    target_y, _ = librosa.load(TARGET_PATH, sr=SAMPLE_RATE)
    target_mfcc = get_mfcc(target_y)

    print("target_mfcc shape =", target_mfcc.shape)
    print("Listening /respeaker/audio_path ...")
    print("Play bell several times, speak several times, then compare distance.")
    print("Log file:", LOG_PATH)

    rospy.Subscriber("/respeaker/audio_path", String, callback_path)
    rospy.Subscriber("/voice/text", Voice, callback_voice)

    rospy.spin()