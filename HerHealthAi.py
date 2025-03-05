import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import os  # Import os for environment variables

# Groq API Configuration
# API Key from Environment Variable - Securely managed
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Please set the GROQ_API_KEY environment variable in your Streamlit secrets.")
    st.stop()  # Stop execution if API key is missing
API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Updated Sample dataset generation to include woman-specific features
def generate_sample_data():
    np.random.seed(42)
    return pd.DataFrame({
        'Age': np.random.randint(18, 60, 100),
        'BMI': np.random.uniform(18.5, 35, 100),
        'Exercise_Hours': np.random.uniform(0, 10, 100),
        'Chronic_Conditions': np.random.randint(0, 5, 100),
        'Menopause': np.random.choice([0, 1], 100), # 1 for yes, 0 for no
        'Pregnancy': np.random.choice([0, 1], 100), # 1 for yes, 0 for no
        'Breastfeeding': np.random.choice([0, 1], 100) # 1 for yes, 0 for no
    })

data = generate_sample_data()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Hierarchical Clustering
linked = linkage(scaled_data, method='ward')
data["Cluster_HC"] = fcluster(linked, 3, criterion='maxclust')

# Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=3, random_state=42)
data["Cluster_GMM"] = gmm.fit_predict(scaled_data)

# Streamlit App
st.set_page_config(page_title="HerHealth AI", page_icon="🩺",)

# Custom CSS to set background image
st.markdown(
    """
    <style>
    body , st.App {
        background-image: url("https://i.imgur.com/PhfKMQm.png"); 
        background-size: cover; 
        background-repeat: no-repeat; 
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlined App Header
st.title("HerHealth AI: Your Personalized Wellness Companion 👩‍⚕️‍⚕️")
st.subheader("AI-Powered Health Insights Tailored for Women📊📝 ")
st.write("""
Welcome to HerHealth AI – your intelligent guide to understanding and improving your well-being.
This app uses advanced AI clustering to analyze your health profile and provide personalized wellness insights and recommendations.
Enter your health details to discover your unique health profile and receive actionable advice to support a healthier, happier you.🫂❤️‍🩹
""")
st.markdown("---")



# User Input Section - Woman-Specific Features Added
st.header("🩺 Enter Your Health Details")
age = st.slider("👩 Age", 18, 120, 30, help="Your age in years.")
bmi = st.slider("⚖️ BMI (Body Mass Index)", 10.0, 50.0, 25.0, help="Your Body Mass Index (kg/m²).")
exercise = st.slider("🏃‍♀️ Exercise Hours Per Week", 0, 10, 3, help="Hours of exercise per week.")
chronic_conditions = st.slider("💊 Chronic Conditions", 0, 5, 1, help="Number of chronic health conditions.")
menopause = st.radio("🩸 Are you experiencing Menopause?", ["No", "Yes"], help="Are you currently in menopause?")
menopause = 1 if menopause == "Yes" else 0
pregnancy = st.radio("🤰 Are you currently Pregnant?", ["No", "Yes"], help="Are you currently pregnant?")
pregnancy = 1 if pregnancy == "Yes" else 0
breastfeeding = st.radio("🤱 Are you currently Breastfeeding?", ["No", "Yes"], help="Are you currently breastfeeding?")
breastfeeding = 1 if breastfeeding == "Yes" else 0

# Input Validation (same)
if not (18 <= age <= 120):
    st.error("Please enter a valid age between 18 and 120 years.")
    st.stop()
if not (10.0 <= bmi <= 50.0):
    st.error("Please enter a valid BMI between 10.0 and 50.0.")
    st.stop()

# Standardizing user input including new features
user_input = pd.DataFrame([[age, bmi, exercise, chronic_conditions, menopause, pregnancy, breastfeeding]],
                          columns=['Age', 'BMI', 'Exercise_Hours', 'Chronic_Conditions', 'Menopause', 'Pregnancy', 'Breastfeeding'])
user_scaled = scaler.transform(user_input)

# Predict clusters for user
hc_cluster = fcluster(linked, 3, criterion='maxclust')[0]
gmm_cluster = gmm.predict(user_scaled)[0]

# Updated Cluster Descriptions - now more informative with woman-specific context
cluster_descriptions_gmm = {
    0: "Profile indicating potential for proactive health management. Likely in good health but could benefit from maintaining a healthy lifestyle.",
    1: "Profile suggesting moderate health risks. May need to focus on lifestyle adjustments, especially diet and exercise, to prevent future health issues.",
    2: "Profile indicating higher health risks. Might benefit from medical consultation and potentially require management of existing or emerging health conditions. Requires attention to lifestyle and medical advice."
}
gmm_cluster_description = cluster_descriptions_gmm.get(gmm_cluster, "Description Not Available")

st.write(f"🔹 **Hierarchical Cluster**: {hc_cluster}")
st.write(f"🔹 **Gaussian Mixture Cluster**: {gmm_cluster} - *{gmm_cluster_description}*")
st.markdown("---")

# Visualizations - Displayed directly (not collapsible)
st.header("📊 Visualizations of Health Data Clusters")

st.subheader("🔍 Hierarchical Clustering Dendrogram")
fig_dendro, ax_dendro = plt.subplots(figsize=(8, 4))
dendrogram(linked, ax=ax_dendro, truncate_mode='level', p=5)
ax_dendro.set_title("Dendrogram (Hierarchical Clustering)")
ax_dendro.set_xlabel("Data Points")
ax_dendro.set_ylabel("Distance")
st.pyplot(fig_dendro)

st.subheader("📊 Cluster Distribution (GMM)")
fig_scatter, ax_scatter = plt.subplots(figsize=(8, 4))
sns.scatterplot(x=data['Age'], y=data['BMI'], hue=data['Cluster_GMM'], palette="Set1", ax=ax_scatter)
ax_scatter.set_title("Age vs BMI (Clustered by GMM)")
st.pyplot(fig_scatter)

st.subheader("📌 Cluster Distribution (Hierarchical)")
fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
data["Cluster_HC"].value_counts().plot(kind="pie", autopct="%1.1f%%", colors=["lightblue", "lightcoral", "lightgreen"], ax=ax_pie)
ax_pie.set_ylabel("")
st.pyplot(fig_pie)
st.markdown("---")

# AI RECOMMENDATIONS - Updated Prompt to include woman-specific features
def get_health_recommendations(age, bmi, exercise, chronic_conditions, menopause, pregnancy, breastfeeding, cluster):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    prompt = f"""
You are a compassionate and highly knowledgeable AI health and wellness expert specializing in women's health. Your goal is to provide personalized, actionable, and encouraging recommendations to a patient based on their detailed health profile and assigned health cluster.

        **Patient Health Profile:**
        - Age: {age} years
        - BMI: {bmi}
        - Exercise: {exercise} hrs/week
        - Chronic Conditions: {chronic_conditions}
        - Menopause Status: {'Yes' if menopause else 'No'}
        - Pregnancy Status: {'Yes' if pregnancy else 'No'}
        - Breastfeeding Status: {'Yes' if breastfeeding else 'No'}
        - Assigned Health Cluster: {cluster} (Analyze this cluster to tailor advice)

        **Instructions for AI Recommendations (Women-Specific Focus):**

        1.  **Dietary Plan (Tailored for Women's Health):**
            *   Provide a sample daily meal plan (breakfast, lunch, dinner, snacks) considering women's nutritional needs across different life stages (e.g., menopause, pregnancy, breastfeeding if applicable based on profile).
            *   Highlight key nutrients relevant for women's health, such as calcium, iron, folate, and omega-3s.
            *   Suggest specific food examples that support hormonal balance and overall well-being in women keeping in mind the indian diet and breakfast options.

        2.  **Exercise Regimen (Appropriate for Women):**
            *   Recommend types of cardio exercises suitable for women at different ages and life stages.
            *   Suggest strength training exercises focusing on areas important for women's health (bone density, muscle strength).
            *   Provide recommendations on exercise frequency, duration, and intensity, considering potential life stage factors like pregnancy or menopause.

        3.  **Lifestyle Modifications (Women's Wellness):**
            *   Offer practical tips for managing stress and promoting mental well-being, relevant to women's common stressors.
            *   Suggest lifestyle adjustments to support hormonal balance and manage symptoms related to menopause or menstrual cycles, if applicable.
            *   Provide advice on sleep hygiene and its importance for women's health.

        4.  **Preventative Health and Medical Advice (Women-Focused):**
            *   Suggest age-appropriate and women-specific health screenings and check-ups (mammograms, Pap smears, bone density scans, etc.), considering guidelines for different age groups and risk factors.
            *   Advise on specific vaccinations relevant for women's health.
            *   Recommend when specialist consultation (e.g., gynecologist, women's health specialist) is appropriate.
            *   **Crucially, state that this is not a substitute for professional medical advice.**

        **Desired Tone:** Empathetic, encouraging, and action-oriented, with a focus on women's unique health needs and life stages. Focus on positive and achievable steps the patient can take to improve their well-being.

        **Important Disclaimer:**  Remind the user that these recommendations are AI-generated and should not replace advice from a qualified healthcare professional, especially for women's specific health concerns.
        """

    payload = {"model": "mixtral-8x7b-32768", "messages": [{"role": "user", "content": prompt}], "temperature": 0.7, "max_tokens": 1500}
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        ai_advice = response.json()["choices"][0]["message"]["content"]
        return ai_advice
    except requests.exceptions.RequestException as e:
        return f"❌ Error fetching AI advice: {e}"

st.header("👩‍⚕️ AI Health Recommendations")
if st.button("Get Personalized AI Advice"):
    with st.spinner("Generating personalized health advice..."):
        recommendations = get_health_recommendations(age, bmi, exercise, chronic_conditions,  menopause, pregnancy, breastfeeding, cluster= gmm_cluster)
    if recommendations:
        st.success("Your Personalized Health Recommendations:")
        for line in recommendations.split('\n'):
            if line.strip().startswith("**"):
                st.markdown(line.strip())
            elif line.strip().startswith("*"):
                st.write("- " + line.strip()[1:].strip())
            else:
                st.write(line.strip())
    else:
        st.error("Could not retrieve AI recommendations.")

st.markdown("---")
st.write("Disclaimer: HerHealth AI provides AI-generated health information for educational purposes only and is not a substitute for professional medical advice. Consult with a healthcare provider for any health concerns.")


#  **Sidebar Information Sections - Added here**
with st.sidebar:
    st.header("📒 General Health Guidelines")
    with st.expander("🩺 Medical Tests for Women", expanded=False):
        st.write("- **Weight:** Regular monitoring.")
        st.write("- **Blood Pressure and Blood Sugar:** Regular checks.")
        st.write("- **Cholesterol Profile:** Every 5 years from age 20.")
        st.write("- **Women's Health:** Clinical breast exams, pelvic exams, Pap smears to protect from cancer.")
        st.write("- **Breast Exams and Bone density screening:** Annual mammograms and bone density screening after 40 is crucial as risk of strokes and osteoporesis increases.")

    with st.expander("❤️ Maintaining Sexual Health", expanded=False):
        st.write("- Use barrier methods like condoms consistently to protect against sexually transmitted infections (STIs)")
        st.write("- Find suitable birth control/family planning.")
        st.write("- Regular STI screenings, Pap smears, pelvic exams.")
        st.write("- Address issues like low libido,inability to reach orgasm, reduced response to sexual stimulation, inadequate lubrication, and painful intercourse etc.")

    with st.expander("🧘 Stress Management", expanded=False):
        st.write("- Excessive stress can cause various health problems like upset stomach or other gastrointestinal issues Back pain, Sleeping difficulties ,Abdominal weight gain etc.")
        st.write("- Incorporate daily practices like deep breathing exercises, meditation, or yoga to calm your mind and reduce stress hormones.")
        st.write("- Aim for 7-9 hours of quality sleep each night with a fixed sleep schedule")
        st.write("- Strong social connections are vital for emotional well-being and resilience. 🫂")

    with st.expander("🤰 Pregnancy and Postpartum", expanded=False):
        st.write("- Increased vaginal discharge is normal in pregnancy but it should not be fishy-smelling or gray discharge as it can be a symptom of Bacterial Vaginosis.")
        st.write("- Be aware of yeast infection signs like vaginal discharge that resembles cottage cheese and smells yeasty.")
        st.write("- Untreated BV is linked to Pre-term labor, Low birth weight, and Miscarriage . So immediate doctor comsultation is advised")
        st.write("- Vaginal bleeding in pregnancy needs immediate attention as in some cases, it also may be a sign of miscarriage.")
        st.write("- Breastfeeding is crucial for infant and mother health as woman who women who breastfeed also have a reduced risk of breast and ovarian cancers.")

    with st.expander("🩸 Menstrual Health and Menopause", expanded=False):
        st.write("- Menstrual health is defined as complete physical, mental, social well-being of woman in relation to her menstrual cycle.")
        st.write("- Conviniently available Sanitary Napkins are harmful to your health.They carry BPA and other chemicals which can cause cancer over time and can interfere with the reproductive system as well.")
        st.write("- The presence of pesticides and herbicides in pads can directly enter your bloodstream to affect your internal organs.To prevent the smell of menstrual blood sanitary napkins are equipped with deodorants and fragrances which can cause infertility.")
        st.write("- Menstrual cups are a safer alternative to sanitary napkins and are sustainable for a longer period of use.")
        st.write("- Menopause symptoms like hot flashes, irregular periods, urinary urgency & infections, Lack of energy, Joint and back pain, Breast tenderness & swelling, Memory & Concentration problems etc.")
        st.write("- Menopause increases the risk of osteoporosis. Ensure adequate calcium and vitamin D intake and engage in weight-bearing exercises. Discuss bone density screening with your doctor and consider supplements if recommended. 🦴.")

    


# --- NEW SECTION FOR CONSULTATION/BOOKING ---
st.markdown("---") # Separator line
st.header("✍️🩺 Consultation & Clinic Access") # Header for the new section

st.write("Need to discuss your health profile in more detail? Request a consultation with a healthcare professional.")
col1, col2 = st.columns(2) # Create columns for layout

with col1:
    consult_name = st.text_input("Your Name")
    consult_email = st.text_input("Your Email")

with col2:
    consult_date = st.date_input("Preferred Date")
    consult_time = st.time_input("Preferred Time")

if st.button("Request Consultation ✉️"):
    if consult_name and consult_email:
        st.success(f"Consultation request sent for {consult_name} ({consult_email}) for {consult_date} at {consult_time}. We will be in touch soon.")
        # In a real app, you would trigger an email sending or store this request in a database.
    else:
        st.warning("Please enter your Name and Email to request a consultation.")

st.subheader("Book an Appointment with a Doctor")
doctor_specialty = st.selectbox("Select Specialty", ["General Physician", "Gynecologist", "Endocrinologist", "Cardiologist", "Dermatologist"]) # Example specialties
doctor_name_pref = st.text_input("Preferred Doctor's Name (Optional)")
appt_date = st.date_input("Appointment Date")

if st.button("Book Appointment 📅"):
    st.success(f"Appointment booking requested for {doctor_specialty} on {appt_date}. We will confirm availability.")
    # In a real app, you would integrate with a doctor booking system here.

st.subheader("🏥 Nearest Clinics Location")
st.write("Find nearby clinics that specialize in women's health services.")

# --- Basic Map using st.map ---
clinic_data = pd.DataFrame({ # Placeholder clinic data - REPLACE with real data source!
    'lat': [34.0522, 34.0530, 34.0490], # Example latitudes near Los Angeles
    'lon': [-118.2437, -118.2500, -118.2350], # Example longitudes
    'clinic_name': ["Women's Health Clinic A", "Family Wellness Center", "City Medical Group"] # Example clinic names
})
st.map(clinic_data,
       latitude='lat',
       longitude='lon',
       size=10) # size parameter controls marker size (approximate)

st.markdown("---")
st.write("Disclaimer: This app provides AI-generated health recommendations for informational purposes only and does not constitute medical advice. Always consult with a qualified healthcare professional for diagnosis and treatment.")

