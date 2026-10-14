import streamlit as st
import joblib
import numpy as np

# Set Page Configuration
st.set_page_config(
    page_title="PriceWise AI — Smart Price Prediction Chatbot",
    page_icon="💻",
    layout="centered"
)

# Custom Header styling
st.title("💻 PriceWise AI")
st.subheader("Smart Used Laptop Price Prediction Chatbot")
st.write("Enter your laptop's specifications below to get an estimated resale price instantly!")

# Load Model and Label Encoders
@st.cache_resource
def load_model_data():
    return joblib.load('price_wise_model.pkl')

try:
    model_data = load_model_data()
    model = model_data['model']
    encoders = model_data['label_encoders']
except Exception as e:
    st.error(f"Error loading the model: {e}. Please make sure 'price_wise_model.pkl' is in the same directory.")
    st.stop()

# Sidebar for helpful info
st.sidebar.header("About PriceWise AI")
st.sidebar.info(
    "This AI Model uses a Random Forest Regressor trained on Indian used laptop market data "
    "to calculate resale values based on brand, age, condition, and key specifications."
)

# Create a Form for User Inputs
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        brand = st.selectbox("Laptop Brand", options=['Apple', 'Dell', 'HP', 'Lenovo', 'Asus'])
        ram = st.selectbox("RAM Size (GB)", options=[4, 8, 16, 32], index=1)
        storage = st.selectbox("Storage Size (GB)", options=[256, 512, 1024], index=1)
        
    with col2:
        age = st.slider("Age of Laptop (Years)", min_value=0.5, max_value=6.0, value=2.0, step=0.1)
        condition = st.selectbox("Overall Condition", options=['Excellent', 'Good', 'Fair', 'Poor'], index=1)
        
    submit_button = st.form_submit_button("Predict Resale Price 🚀")

# Handle prediction on form submit
if submit_button:
    try:
        # Preprocess / Encode the inputs
        encoded_brand = encoders['Brand'].transform([brand])[0]
        encoded_condition = encoders['Condition'].transform([condition])[0]
        
        # Prepare input features
        features = [[encoded_brand, ram, storage, age, encoded_condition]]
        
        # Predict value
        prediction = model.predict(features)[0]
        
        # Generate a reasonable price range (lower and upper bound)
        lower_bound = int(round(prediction * 0.9, -2))
        upper_bound = int(round(prediction * 1.1, -2))
        
        # Display Result in chatbot-style container
        st.success("🔮 Prediction Ready!")
        st.chat_message("assistant").write(
            f"💬 **PriceWise AI:** Mee laptop details batti estimated resale price "
            f"**Rs. {lower_bound:,} - Rs. {upper_bound:,}** undavachu."
        )
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")