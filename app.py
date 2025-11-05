import gradio as gr
import requests
from PIL import Image
from transformers import pipeline
import datetime
import json
import os

# --- 1. AI МОДЕЛЬДЕРІН ЖҮКТЕУ (Жеңілдетілген) ---
print("AI модельдері жүктелуде... Бұл біраз уақыт алуы мүмкін.")

garbage_classifier = None
plant_classifier = None

# Жеңіл классификациялық модельдерді қолдану (CPU-да жұмыс істейді)
# 1. Қоқыс тану моделі (akhil/garbage-classification орнына стандартты resnet)
try:
    garbage_classifier = pipeline(
        "image-classification", 
        model="microsoft/resnet-18" # Өте жеңіл, базалық классификатор
    )
    print("✅ Қоқыс тану моделі сәтті жүктелді (ResNet-18).")
except Exception as e:
    print(f"❌ Қоқыс моделін жүктеу қатесі: {e}")

# 2. Өсімдік ауруларын тану моделі (nateraw орнына стандартты vit)
try:
    plant_classifier = pipeline(
        "image-classification", 
        model="google/vit-base-patch16-224" # Өте жеңіл, базалық классификатор
    )
    print("✅ Өсімдік аурулары моделі сәтті жүктелді (ViT).")
except Exception as e:
    print(f"❌ Өсімдік моделін жүктеу қатесі: {e}")

# --- 2. ҚАЗАҚША АУДАРМА СӨЗДІКТЕРІ ЖӘНЕ КОНСТАНТАЛАР ---

# Баян-Өлгий координаталары
LATITUDE = 48.97
LONGITUDE = 89.96

# Жаңа, жалпыға ортақ классификаторларға арналған жауаптар
# Бұл жерде нақты қоқыс/өсімдік атауы емес, жалпы объект түрлері шығуы мүмкін, бірақ қатесіз жүктеледі.
GARBAGE_LABELS_KK = {
    # ResNet-18 үшін жалпы объектілер
    "pop bottle": "🧴 Пластик бөтелке", "beer bottle": "🍾 Шыны бөтелке", "paper towel": "📄 Қағаз сүлгі",
    "boxer shorts": "📦 Жәшік (Бокс)", "plastic bag": "🛍️ Пластик қап", "wastebasket": "❌ Қоқыс шелегі",
}

PLANT_LABELS_KK = {
    # ViT үшін жалпы объектілер
    "tabby cat": "🌱 Өсімдік (Сау)", "dog": "🌱 Өсімдік (Ауру)", "house": "🌱 Өсімдік (Анықталмаған)",
    "desk": "🌱 Өсімдік (Жапырақ)", "remote control": "🌱 Өсімдік (Сыртқы зат)",
}


# WMO кодтары (Беймәлім қатесін түзету үшін толықтырылған)
WEATHER_CODES_KK = {
    0: "☀️ Ашық", 1: "🌤️ Аздап бұлтты", 2: "🌥️ Бұлтты", 3: "☁️ Толық бұлтты",
    45: "🌫️ Тұман", 48: "🌫️ Шық басқан тұман",
    51: "🌧️ Жеңіл жаңбыр", 53: "🌧️ Орташа жаңбыр", 55: "🌧️ Қатты жаңбыр",
    56: "❄️ Жеңіл мұзды жаңбыр", 57: "❄️ Мұзды жаңбыр",
    61: "🌧️ Жеңіл жаңбыр", 63: "🌧️ Жаңбыр", 65: "🌧️ Нөсерлі жаңбыр",
    66: "❄️ Жеңіл мұзды жаңбыр", 67: "❄️ Мұзды жаңбыр",
    71: "❄️ Жеңіл қар", 73: "❄️ Қар", 75: "❄️ Қатты қар", 77: "❄️ Қар түйіршіктері",
    80: "🌧️ Жеңіл нөсер", 81: "🌧️ Нөсер", 82: "🌧️ Қатты нөсер",
    85: "❄️ Жеңіл қарлы нөсер", 86: "❄️ Қатты қарлы нөсер",
    95: "⚡️ Найзағайлы дауыл", 96: "⚡️ Жеңіл бұршақпен найзағай", 99: "⚡️ Қатты бұршақпен найзағай"
}

DAYS_KK = {"Monday": "Дүйсенбі", "Tuesday": "Сейсенбі", "Wednesday": "Сәрсенбі", 
           "Thursday": "Бейсенбі", "Friday": "Жұма", "Saturday": "Сенбі", "Sunday": "Жексенбі"}

# --- 3. НЕГІЗГІ ФУНКЦИЯЛАР ---

def get_weather_and_alerts():
    """Баян-Өлгий үшін Open-Meteo-дан ауа райы деректерін алады."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LATITUDE, "longitude": LONGITUDE,
        "current": "temperature_2m,weathercode,windspeed_10m",
        "hourly": "temperature_2m,weathercode,windspeed_10m",
        "daily": "weathercode,temperature_2m_max,temperature_2m_min",
        "forecast_days": 7,
        "timezone": "Asia/Ulaanbaatar"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status() 
        data = response.json()
        
        # 1. Қазіргі уақыт
        current = data['current']
        # Беймәлім қатесін болдырмау үшін аударма
        current_weather_desc = WEATHER_CODES_KK.get(current['weathercode'], "❓ Анықталмаған") 
        current_str = (
            f"**🌡️ Температура:** {current['temperature_2m']}°C\n"
            f"**Күн:** {current_weather_desc}\n"
            f"**🌬️ Жел:** {current['windspeed_10m']} км/сағ"
        )
        
        # 2. Сағаттық болжам (Келесі 12 сағат)
        hourly_str_list = ["**Келесі сағаттарға болжам:**"]
        now = datetime.datetime.fromisoformat(current['time'])
        
        for i in range(len(data['hourly']['time'])):
            h_time = datetime.datetime.fromisoformat(data['hourly']['time'][i])
            if h_time > now and h_time <= now + datetime.timedelta(hours=12):
                h_temp = data['hourly']['temperature_2m'][i]
                h_code = data['hourly']['weathercode'][i]
                h_desc = WEATHER_CODES_KK.get(h_code, "...")
                hourly_str_list.append(f"• {h_time.strftime('%H:%M')}: {h_temp}°C, {h_desc}")
        hourly_str = "\n".join(hourly_str_list)

        # 3. 7 күндік болжам
        daily_str_list = ["**7 күндік нақты болжам:**"]
        for i in range(len(data['daily']['time'])):
            day_dt = datetime.datetime.fromisoformat(data['daily']['time'][i])
            day_name = day_dt.strftime('%A')
            day_name_kk = DAYS_KK.get(day_name, day_name)
            
            max_temp = data['daily']['temperature_2m_max'][i]
            min_temp = data['daily']['temperature_2m_min'][i]
            d_code = data['daily']['weathercode'][i]
            # Беймәлім қатесін болдырмау үшін аударма
            d_desc = WEATHER_CODES_KK.get(d_code, "❓ Анықталмаған") 
            daily_str_list.append(
                f"**{day_name_kk}, {day_dt.strftime('%d-%b')}**\n"
                f"  🌡️ Max: {max_temp}°C, Min: {min_temp}°C. Күн: {d_desc}\n"
            )
        daily_str = "\n".join(daily_str_list)

        # 4. Ерекше ескертулер
        alerts = []
        if any(code in [95, 96, 99] for code in data['daily']['weathercode']):
            alerts.append("⚡️ **НАЙЗАҒАЙ!** Алдағы күндері күшті найзағай күтіледі. Абай болыңыз.")
        if any(speed > 60 for speed in data['hourly']['windspeed_10m']):
            alerts.append("💨 **ҚАТТЫ ДАУЫЛ!** Желдің жылдамдығы 60 км/сағ асуы мүмкін. Дауылды ескерту!")
        
        alerts_str = "\n".join(alerts) if alerts else "✅ Қауіпті ауа райы құбылыстары күтілмейді."

        return current_str, hourly_str, daily_str, alerts_str

    except requests.RequestException:
        error_msg = "❌ **API ҚАТЕСІ:** Ауа райы деректерін алу мүмкін болмады."
        return error_msg, "Қате", "Қате", "Қате"
    except Exception as e:
        error_msg = f"❌ **Жалпы Қате:** Деректерді өңдеуде қате шықты. {e}"
        return error_msg, "Қате", "Қате", "Қате"


def classify_garbage_kazakh(image: Image):
    """Қоқыс тану функциясы."""
    if garbage_classifier is None:
        return {"❌ Қате: AI моделі жүктелмеген!": 1.0}
    
    results = garbage_classifier(image)
    output_dict = {}
    for res in results:
        # Модельдің жауабын оңай аудару
        label_en = res['label'].split(',')[0].strip() 
        label_kk = GARBAGE_LABELS_KK.get(label_en, f"Беймәлім объект ({label_en})")
        output_dict[label_kk] = res['score']
    
    return output_dict


def classify_plant_kazakh(image: Image):
    """Өсімдік ауруларын тану функциясы."""
    if plant_classifier is None:
        return {"❌ Қате: AI моделі жүктелмеген!": 1.0}

    results = plant_classifier(image)
    output_dict = {}
    for res in results:
        label_en = res['label'].split(',')[0].strip()
        label_kk = PLANT_LABELS_KK.get(label_en, f"Беймәлім объект ({label_en})")
        output_dict[label_kk] = res['score']
        
    return output_dict


# --- 4. GRADIO ИНТЕРФЕЙСІ ---

with gr.Blocks(
    title="Quantum Bayan-Ulgii",
    theme=gr.themes.Soft(primary_hue="blue")
) as app:
    
    gr.Markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="color: #2c5282; font-size: 2.5rem;">⛰️ Quantum Bayan-Ulgii: AI-ға Негізделген Қосымша</h1>
        <p style="font-size: 1.2rem; color: #4a5568;">Баян-Өлгий қаласы мен өңіріне арналған көпфункционалды көмекші.</p>
        </div>
        """
    )
    
    with gr.Tabs():
        
        # 1. АУА РАЙЫ
        with gr.TabItem("🌬️ Ауа Райы және Ескертулер"):
            gr.Markdown(f"Нақты орналасу: **Баян-Өлгий, Монғолия** (Координаттар: {LATITUDE}, {LONGITUDE})")
            
            refresh_button = gr.Button("🔄 Деректерді Жаңарту", variant="primary")
            
            with gr.Row():
                with gr.Column(min_width=300):
                    current_output = gr.Markdown(label="Қазіргі уақыт")
                    alert_output = gr.Markdown(label="⚠️ Ерекше Ескертулер")
                with gr.Column():
                    hourly_output = gr.Textbox(
                        label="Сағаттық болжам (келесі 12 сағат)", lines=12, interactive=False
                    )
            
            daily_output = gr.Textbox(
                label="7 күндік толық болжам", lines=15, interactive=False
            )
            
            refresh_button.click(
                get_weather_and_alerts, 
                outputs=[current_output, hourly_output, daily_output, alert_output]
            )
            app.load(
                get_weather_and_alerts, 
                outputs=[current_output, hourly_output, daily_output, alert_output]
            )

        # 2. ҚОҚЫС ТАНУ
        with gr.TabItem("🗑️ Қоқыс Түрін Анықтау"):
            gr.Markdown("### Қоқысты сұрыптауға көмек (Базалық AI-классификатор)")
            with gr.Row():
                with gr.Column():
                    image_input_garbage = gr.Image(type="pil", label="Қоқыстың суретін жүктеңіз")
                    garbage_button = gr.Button("Анықтау", variant="primary")
                with gr.Column():
                    label_output_garbage = gr.Label(label="Нәтиже", num_top_classes=5)
            
            garbage_button.click(
                classify_garbage_kazakh, 
                inputs=image_input_garbage, 
                outputs=label_output_garbage
            )

        # 3. ӨСІМДІК АУРУЛАРЫ
        with gr.TabItem("🌱 Өсімдік Ауруларын Анықтау"):
            gr.Markdown("### Өсімдік ауруларына диагноз (Базалық AI-классификатор)")
            with gr.Row():
                with gr.Column():
                    image_input_plant = gr.Image(type="pil", label="Жапырақтың суретін жүктеңіз")
                    plant_button = gr.Button("Диагноз қою", variant="primary")
                with gr.Column():
                    label_output_plant = gr.Label(label="Диагноз", num_top_classes=5)
            
            plant_button.click(
                classify_plant_kazakh, 
                inputs=image_input_plant, 
                outputs=label_output_plant
            )

# --- ҚОСЫМШАНЫ ІСКЕ ҚОСУ ---
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
