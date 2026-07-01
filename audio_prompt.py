import requests

# 伺服器 IP
BASE_URL = "http://192.168.103.14:8888"


def send_media_task(endpoint, file_path, prompt, file_key):
    try:
        with open(file_path, 'rb') as f:
            files = {file_key: f}
            data = {'prompt': prompt}
            #print(f"[*] 正在傳送 {file_path} 到 {endpoint}...")

            response = requests.post(f"{BASE_URL}{endpoint}", files=files, data=data, timeout=60)

        if response.status_code == 200:
            return response.json().get('ans')
        else:
            return f"伺服器錯誤: {response.text}"
    except Exception as e:
        return f"連線失敗: {e}"


if __name__ == "__main__":
    audio_file = r"C:\Users\rayso\Desktop\python\recording_2s.wav"
    audio_prompt = """You are an expert order-taking assistant. The provided audio is a recording of a customer placing an order,
                but it contains a lot of noisy background sounds. Please perform the following tasks:
                1. Ignore all background noise, wind, music, or chimes.
                2. Listen closely to the human voice and transcribe exactly what the customer said (English).
                answer it in ths format for example: ["apple","coffee"]
                just answer me with a python list
                if the order is two apple, please answer me in ["apple", "apple"]
                if nothing is in the audio answer: ["nothing'], don't give me extral text
                Reply with only one of these two exact phrases:
                1. ["nothing']
                2. ["item1", "item2"]"""

    print("\n--- audio ---")
    audio_ans = send_media_task("/Fambot", audio_file, audio_prompt, "audio")
    print(f"[AI response]: {audio_ans}")

    image_file = r"C:\Users\rayso\Downloads\guest.jpg"
    image_prompt = """
    Describe the main person in the image.

    Return JSON only:
    {
      "hat": false,
      "glasses": false,
      "upper_color": "blue",
      "skin_tone": "light",
      "hair_length": "short",
      "age_group": "adult"
    }

    Rules:
    - Output JSON only.
    - Use one-word colors for upper_color.
    - skin_tone must be one of: light, medium, dark, unclear.
    - age_group must be one of: child, adult, elderly, unclear.
    - If uncertain, use "unclear".
    """
    print("\n--- image ---")
    image_ans = send_media_task("/Vision", image_file, image_prompt, "image")
    print(f"[AI response]: {image_ans}")
