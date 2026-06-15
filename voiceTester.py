import os
import time
import base64
import json
import sounddevice as sd
import numpy as np
import soundfile as sf
import requests

# ================= 配置設定 =================
API_KEY = "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"
MODEL = "xiaomi/mimo-v2.5"
AUDIO_PATH = r"C:\Users\rayso\Desktop\python\recording_2s.wav"

SAMPLE_RATE = 44100
RECORD_SECONDS = 5  # 聽訂單建議可以改長一點，例如 5 秒或 10 秒
DEVICE_ID = 1


# ================= 功能函式 =================
def encode_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode('utf-8')


def call_ai_api():
    print(f"\n[AI] 正在分析訂單錄音檔...")
    try:
        base64_audio = encode_audio(AUDIO_PATH)

        # --- 這裡幫你重寫了優化過的 AI Prompt ---
        optimized_prompt = (
            "You are an expert order-taking assistant. The provided audio is a recording of a customer placing an order, "
            "but it contains a lot of noisy background sounds. Please perform the following tasks:\n"
            "1. Ignore all background noise, wind, music, or chimes.\n"
            "2. Listen closely to the human voice and transcribe exactly what the customer said (in the language they spoke, e.g., Cantonese, English, or Mandarin).\n"
            "3. Clearly list what the customer NEEDS (items, quantity, special requests, or questions they asked).\n"
            "Provide the output in a clear, bulleted format."
        )

        payload = {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": optimized_prompt
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64_audio,
                                "format": "wav"
                            }
                        }
                    ]
                }
            ]
        }

        start_time = time.time()
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=20
        )
        end_time = time.time()

        result = response.json()
        print("\n" + "=" * 40)
        if 'choices' in result:
            content = result['choices'][0]['message']['content']
            print(f"[AI 訂單分析結果]:\n{content}")
            print("-" * 40)
            print(f"[API 執行時間]: {end_time - start_time:.2f} 秒")
        else:
            print("[AI 錯誤]: API 回傳異常")
            print(json.dumps(result, indent=2))
        print("=" * 40 + "\n")

    except Exception as e:
        print(f"[AI 發生錯誤]: {e}")


# ================= 主程式 =================
if __name__ == "__main__":
    print(sd.query_devices())
    os.makedirs(os.path.dirname(AUDIO_PATH), exist_ok=True)

    print(f"使用設備 ID: {DEVICE_ID}")
    print("\n準備錄音...")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    print(f"🔴 錄音中 (預計錄製 {RECORD_SECONDS} 秒)... 請讓客戶說話")

    try:
        recording = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE),
                           samplerate=SAMPLE_RATE,
                           channels=1,
                           device=DEVICE_ID,
                           dtype='float32')
        sd.wait()
        print("⏹️ 錄音結束！")

        sf.write(AUDIO_PATH, recording, SAMPLE_RATE)
        call_ai_api()

    except Exception as e:
        print(f"\n執行失敗: {e}")
