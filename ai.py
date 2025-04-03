import streamlit as st
import easyocr
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# تنظیمات صفحه Streamlit
st.set_page_config(page_title="Cryptocurrency Chart Analyzer", page_icon="📈")

# عنوان صفحه
st.title("Cryptocurrency Chart Analyzer")
st.write("Upload a chart image of a cryptocurrency, and the system will extract the text and analyze the trends.")

# بارگذاری تصویر
uploaded_file = st.file_uploader("Upload an image of the cryptocurrency chart", type=["png", "jpg", "jpeg"])

# تحلیل تصویر
if uploaded_file is not None:
    # نمایش تصویر بارگذاری شده
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Chart", use_column_width=True)

    # استخراج متن با استفاده از EasyOCR
    reader = easyocr.Reader(['en'])
    img_array = np.array(image)
    result = reader.readtext(img_array)

    # نمایش متن استخراج شده
    st.subheader("Extracted Text")
    extracted_text = ' '.join([item[1] for item in result])
    st.write(extracted_text)

    # شبیه‌سازی داده‌های قیمت از متن استخراج‌شده (برای تست، این قسمت قابل ارتقا است)
    prices = [float(word) for word in extracted_text.split() if word.replace('.', '', 1).isdigit()]
    
    if len(prices) >= 14:
        # تحلیل Williams %R
        period = 14
        high = max(prices[-period:])
        low = min(prices[-period:])
        close = prices[-1]
        williams_r = ((high - close) / (high - low)) * -100
        
        st.subheader("Trend Analysis - Williams %R")
        st.write(f"Williams %R: {williams_r:.2f}")
        
        if williams_r > -20:
            st.write("🔴 وضعیت اشباع خرید → احتمال کاهش قیمت → **فروش توصیه می‌شود.**")
            st.write(f"📉 نقطه فروش پیشنهادی: {close:.2f}")
        elif williams_r < -80:
            st.write("🟢 وضعیت اشباع فروش → احتمال افزایش قیمت → **خرید توصیه می‌شود.**")
            st.write(f"📈 نقطه خرید پیشنهادی: {close:.2f}")
        else:
            st.write("⚪ روند خنثی → منتظر سیگنال قوی‌تر باش.")
        
        # تحلیل RSI (شاخص قدرت نسبی)
        st.subheader("Trend Analysis - RSI (Relative Strength Index)")
        gains = [prices[i] - prices[i-1] for i in range(1, len(prices)) if prices[i] > prices[i-1]]
        losses = [prices[i-1] - prices[i] for i in range(1, len(prices)) if prices[i] < prices[i-1]]
        
        avg_gain = np.mean(gains[-14:]) if gains else 0
        avg_loss = np.mean(losses[-14:]) if losses else 0
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        
        st.write(f"RSI: {rsi:.2f}")
        if rsi > 70:
            st.write("🔴 اشباع خرید → احتمال کاهش قیمت → **فروش توصیه می‌شود.**")
        elif rsi < 30:
            st.write("🟢 اشباع فروش → احتمال افزایش قیمت → **خرید توصیه می‌شود.**")
        else:
            st.write("⚪ روند خنثی → منتظر سیگنال قوی‌تر باش.")
        
        # تحلیل MACD (Moving Average Convergence Divergence)
        st.subheader("Trend Analysis - MACD")
        short_period = 12
        long_period = 26
        signal_period = 9
        
        short_ema = np.mean(prices[-short_period:])
        long_ema = np.mean(prices[-long_period:])
        macd = short_ema - long_ema
        
        signal = np.mean([macd] * signal_period)  # سیگنال MACD
        macd_histogram = macd - signal
        
        st.write(f"MACD: {macd:.2f}, Signal: {signal:.2f}, Histogram: {macd_histogram:.2f}")
        if macd > signal:
            st.write("🔵 سیگنال خرید → **خرید توصیه می‌شود.**")
        elif macd < signal:
            st.write("🔴 سیگنال فروش → **فروش توصیه می‌شود.**")
        else:
            st.write("⚪ روند خنثی → منتظر سیگنال قوی‌تر باش.")
        
    else:
        st.write("❌ اطلاعات کافی برای تحلیل موجود نیست.")

    # نمایش نمودار روند قیمتی
    if prices:
        st.subheader("Price Trend Chart")
        plt.figure(figsize=(8, 4))
        plt.plot(prices, marker='o', linestyle='-', color='b')
        plt.title("Cryptocurrency Price Trend")
        plt.xlabel("Time")
        plt.ylabel("Price")
        st.pyplot(plt)
