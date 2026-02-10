import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sklearn

# Check sklearn version
st.sidebar.write(f"scikit-learn version: {sklearn.__version__}")

# Set page config
st.set_page_config(
    page_title="Fitness Calorie Burn Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Model
@st.cache_resource
def load_model():
    try:
        # Try loading with joblib
        model = joblib.load('burns_calories_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model with joblib: {e}")
        # Try alternative loading method
        import pickle
        with open('burns_calories_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.error("Please ensure 'burns_calories_model.pkl' is in the repository and was trained with scikit-learn 1.5.2")
    st.stop()

# Title and Intro
st.title("Fitness Calorie Burn Predictor")
st.markdown("""
This application predicts the **Calorie Burn Category** (Low, Medium, High, Very High) based on your personal stats, workout details, and nutritional intake.
Adjust the parameters below to see the prediction!
""")

# --- Input Sections ---

# Group 1: Personal Information
st.header("1. Personal Information")
col1, col2, col3, col4 = st.columns(4)

with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=30)
    gender = st.selectbox("Gender", ['Male', 'Female'])

with col2:
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
    height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.75)

with col3:
    fat_percentage = st.number_input("Fat Percentage (%)", min_value=1.0, max_value=60.0, value=20.0)
    water_intake = st.number_input("Water Intake (liters)", min_value=0.0, max_value=10.0, value=2.5)

with col4:
    experience_level = st.slider("Experience Level (1-3)", 1, 3, 2)
    # 1=Beginner, 2=Intermediate, 3=Advanced (Assumption based on data range or usage)

# Derived Personal Stats
bmi = weight / (height ** 2)
lean_mass_kg = weight * (1 - fat_percentage / 100)

# Group 2: Workout Details
st.header("2. Workout Details")
w_col1, w_col2, w_col3 = st.columns(3)

with w_col1:
    workout_type = st.selectbox("Workout Type", ['Strength', 'HIIT', 'Cardio', 'Yoga'])
    session_duration = st.number_input("Session Duration (hours)", 0.2, 5.0, 1.0)
    workout_frequency = st.slider("Workout Frequency (days/week)", 1, 7, 3)

with w_col2:
    body_part = st.selectbox("Body Part", ['Legs', 'Chest', 'Arms', 'Shoulders', 'Abs', 'Back', 'Forearms'])
    target_muscle = st.selectbox("Target Muscle Group", [
        'Shoulders, Triceps', 'Back, Core, Shoulders', 'Quadriceps, Glutes', 'Biceps, Forearms', 
        'Chest, Triceps', 'Core, Obliques', 'Core', 'Back, Biceps', 'Upper Chest, Triceps', 
        'Core, Lower Back', 'Lower Chest, Triceps', 'Core, Shoulders, Hips', 'Rear Deltoids, Upper Back', 
        'Quadriceps, Hamstrings, Glutes', 'Core, Shoulders, Legs', 'Shoulders, Upper Back', 
        'Chest, Triceps, Shoulders', 'Triceps', 'Obliques, Core', 'Full Body', 'Glutes, Hamstrings', 
        'Quadriceps, Calves, Glutes', 'Lower Abs', 'Legs, Shoulders, Core', 'Shoulders', 
        'Upper Back, Rear Deltoids', 'Calves', 'Glutes, Hamstrings, Core', 'Triceps, Chest', 
        'Back, Hamstrings, Glutes', 'Lower Back, Glutes', 'Lower Abs, Hip Flexors', 
        'Full Body, Core, Shoulders', 'Full Core', 'Quadriceps', 'Legs, Core'
    ])
    type_of_muscle = st.selectbox("Type of Muscle", ['Lats', 'Grip Strength', 'Upper', 'Wrist Flexors', 'Lower', 'Middle', 'Lower Chest', 'Triceps', 'Quads', 'Anterior', 'Posterior', 'Wrist Extensors', 'Lateral'])

with w_col3:
    equipment = st.selectbox("Equipment Needed", [
        'Cable Machine', 'Step or Box', 'Parallel Bars or Chair', 'Wall', 'Resistance Band or Cable Machine', 
        'None or Dumbbells', 'Pull-up Bar', 'Barbell', 'Low Bar or TRX', 'Dumbbells', 'Bench or Sturdy Surface', 
        'Bench or Step', 'Cable Machine or Resistance Band', 'Bench or Chair', 'Bench, Barbell', 'Kettlebell', 
        'Resistance Band', 'Box or Platform', 'Dumbbells or Barbell', 'None or Dumbbell'
    ])
    difficulty = st.selectbox("Difficulty Level", ['Advanced', 'Intermediate', 'Beginner'])
    sets = st.number_input("Sets", 1, 10, 3)
    reps = st.number_input("Reps", 1, 50, 12)

# Group 3: Physiological Data
st.header("3. Physiological Data")
p_col1, p_col2, p_col3 = st.columns(3)

with p_col1:
    resting_bpm = st.number_input("Resting BPM", 40.0, 100.0, 60.0)
with p_col2:
    avg_bpm = st.number_input("Avg BPM", 60.0, 200.0, 130.0)
with p_col3:
    max_bpm = st.number_input("Max BPM", 80.0, 220.0, 180.0)

# Derived Physiological Stats
pct_maxhr = avg_bpm / max_bpm if max_bpm > 0 else 0
pct_hrr = (avg_bpm - resting_bpm) / (max_bpm - resting_bpm) if (max_bpm - resting_bpm) > 0 else 0

# Group 4: Nutrition
st.header("4. Nutrition (Daily Intake)")
n_col1, n_col2, n_col3 = st.columns(3)

with n_col1:
    calories = st.number_input("Daily Calories", 1000.0, 5000.0, 2500.0)
    carbs = st.number_input("Carbs (g)", 0.0, 500.0, 300.0)
    proteins = st.number_input("Proteins (g)", 0.0, 500.0, 150.0)
    fats = st.number_input("Fats (g)", 0.0, 200.0, 70.0)
    sugar_g = st.number_input("Sugar (g)", 0.0, 100.0, 30.0)

with n_col2:
    diet_type = st.selectbox("Diet Type", ['Vegan', 'Vegetarian', 'Paleo', 'Keto', 'Low-Carb', 'Balanced'])
    meal_type = st.selectbox("Meal Type", ['Lunch', 'Breakfast', 'Snack', 'Dinner'])
    cooking_method = st.selectbox("Cooking Method", ['Grilled', 'Fried', 'Boiled', 'Baked', 'Steamed', 'Raw', 'Roasted'])
    daily_meals_freq = st.number_input("Daily Meals Frequency", 1, 8, 3)

with n_col3:
    sodium_mg = st.number_input("Sodium (mg)", 0.0, 5000.0, 2000.0)
    cholesterol_mg = st.number_input("Cholesterol (mg)", 0.0, 1000.0, 200.0)
    serving_size_g = st.number_input("Serving Size (g)", 0.0, 1000.0, 300.0)
    prep_time = st.number_input("Prep Time (min)", 0.0, 120.0, 15.0)
    cook_time = st.number_input("Cook Time (min)", 0.0, 120.0, 20.0)
    rating = st.slider("Meal Rating", 0.0, 5.0, 4.0)

# Derived Nutrition Stats
pct_carbs = (carbs * 4) / calories if calories > 0 else 0
protein_per_kg = proteins / weight if weight > 0 else 0
physical_exercise = 1.0 # Placeholder assumption or need input? 
# In dataset, 'Physical exercise' might be a specific metric. 
# Looking at the list, it's 'Physical exercise'. Let's add an input for it.
n_col1_1, n_col1_2 = st.columns(2)
with n_col1_1:
    physical_exercise_val = st.number_input("Physical Exercise Level (activity factor?)", 0.0, 5.0, 1.0)


# --- Prediction ---

if st.button("Predict Calorie Burn Category", type="primary"):
    
    # Construct DataFrame
    input_data = {
        'Age': [age],
        'Gender': [gender],
        'Weight (kg)': [weight],
        'Height (m)': [height],
        'Max_BPM': [max_bpm],
        'Avg_BPM': [avg_bpm],
        'Resting_BPM': [resting_bpm],
        'Session_Duration (hours)': [session_duration],
        'Workout_Type': [workout_type],
        'Fat_Percentage': [fat_percentage],
        'Water_Intake (liters)': [water_intake],
        'Workout_Frequency (days/week)': [workout_frequency],
        'Experience_Level': [experience_level],
        'BMI': [bmi],
        'Daily meals frequency': [daily_meals_freq],
        'Physical exercise': [physical_exercise_val],
        'Carbs': [carbs],
        'Proteins': [proteins],
        'Fats': [fats],
        'Calories': [calories],
        'meal_type': [meal_type],
        'diet_type': [diet_type],
        'sugar_g': [sugar_g],
        'sodium_mg': [sodium_mg],
        'cholesterol_mg': [cholesterol_mg],
        'serving_size_g': [serving_size_g],
        'cooking_method': [cooking_method],
        'prep_time_min': [prep_time],
        'cook_time_min': [cook_time],
        'rating': [rating],
        'Sets': [sets],
        'Reps': [reps],
        'Target Muscle Group': [target_muscle],
        'Equipment Needed': [equipment],
        'Difficulty Level': [difficulty],
        'Body Part': [body_part],
        'Type of Muscle': [type_of_muscle],
        'pct_carbs': [pct_carbs],
        'protein_per_kg': [protein_per_kg],
        'pct_HRR': [pct_hrr],
        'pct_maxHR': [pct_maxhr],
        'lean_mass_kg': [lean_mass_kg]
    }
    
    input_df = pd.DataFrame(input_data)
    
    # Display calculated metrics for user info
    st.info(f"**Calculated Metrics** | BMI: {bmi:.2f} | Lean Mass: {lean_mass_kg:.2f}kg | % Max HR: {pct_maxhr*100:.1f}%")

    try:
        prediction = model.predict(input_df)[0]
        
        # Style the result
        color_map = {
            'Low': 'blue',
            'Medium': 'orange',
            'High': 'red',
            'Very High': 'purple'
        }
        color = color_map.get(prediction, 'gray')
        
        st.markdown("---")
        st.subheader("Prediction Result")
        st.markdown(f"### The predicted Calorie Burn Category is: <span style='color:{color}'>{prediction}</span>", unsafe_allow_html=True)
        
        # Add some fun visualizations or gauges if needed?
        # For now, just the result is good.

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.write("Debug - Input Data Columns:")
        st.write(input_df.columns.tolist())


# Sidebar
with st.sidebar:
    st.header("About")
    st.info("This model uses a **Random Forest Classifier** trained on fitness and lifestyle data.")
    st.write("---")
    st.write("Created with Streamlit")
