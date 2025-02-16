import streamlit as st
from transformers import BertForSequenceClassification, BertTokenizer
from huggingface_hub import hf_hub_download
import torch
import PyPDF2
import pickle
import re
from nltk.corpus import stopwords
import nltk
from io import BytesIO

# Download NLTK stopwords
nltk.download('stopwords')

# Hugging Face Model Repository
HF_MODEL_REPO = "DrSyedFaizan/medReport"

@st.cache_resource
def load_model():
    """Load model, tokenizer, and label encoder from Hugging Face."""
    
    try:
        st.info("🔄 Loading model from Hugging Face...")

        # Download model, tokenizer, and label encoder from Hugging Face
        model = BertForSequenceClassification.from_pretrained(HF_MODEL_REPO)
        tokenizer = BertTokenizer.from_pretrained(HF_MODEL_REPO)

        # Load label encoder
        label_encoder_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="label_encoder.pkl")
        with open(label_encoder_path, "rb") as f:
            label_encoder = pickle.load(f)

        st.success("✅ Model Loaded Successfully!")
        return model, tokenizer, label_encoder

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None

# Load the model once
model, tokenizer, label_encoder = load_model()

# Disease Information Data
disease_data = {
    "Peptic Ulcer Disease": {
        "description": "A sore that develops on the lining of the esophagus, stomach, or small intestine.",
        "medicines": ["Omeprazole", "Pantoprazole", "Ranitidine", "Esomeprazole", "Amoxicillin"],
        "specialists": ["Gastroenterologist", "General Physician", "Internal Medicine Specialist"]
    },
    "Type 2 Diabetes Mellitus": {
        "description": "A chronic condition that affects the way the body processes blood sugar (glucose).",
        "medicines": ["Metformin", "Glipizide", "Insulin", "Sitagliptin", "Canagliflozin"],
        "specialists": ["Endocrinologist", "Diabetologist", "Nutritionist"]
    },
    "Acute Myocardial Infarction": {
        "description": "A medical emergency where the blood flow to the heart is blocked.",
        "medicines": ["Aspirin", "Clopidogrel", "Statins", "Beta Blockers", "ACE Inhibitors"],
        "specialists": ["Cardiologist", "Emergency Medicine Specialist"]
    },
    "Chronic Obstructive Pulmonary Disease": {
        "description": "A group of lung diseases that block airflow and make breathing difficult.",
        "medicines": ["Tiotropium", "Albuterol", "Ipratropium", "Fluticasone", "Salmeterol"],
        "specialists": ["Pulmonologist", "General Physician", "Respiratory Therapist"]
    },
    "Cerebrovascular Accident (Stroke)": {
        "description": "A condition caused by the interruption of blood flow to the brain.",
        "medicines": ["Alteplase", "Aspirin", "Clopidogrel", "Warfarin", "Atorvastatin"],
        "specialists": ["Neurologist", "Rehabilitation Specialist", "Neurosurgeon"]
    },
    "Deep Vein Thrombosis": {
        "description": "A blood clot forms in a deep vein, usually in the legs.",
        "medicines": ["Warfarin", "Heparin", "Apixaban", "Dabigatran", "Rivaroxaban"],
        "specialists": ["Hematologist", "Vascular Surgeon", "Cardiologist"]
    },
    "Chronic Kidney Disease": {
        "description": "The gradual loss of kidney function over time.",
        "medicines": ["Erythropoietin", "Phosphate Binders", "ACE Inhibitors", "Diuretics", "Calcitriol"],
        "specialists": ["Nephrologist", "Dietitian", "Internal Medicine Specialist"]
    },
    "Community-Acquired Pneumonia": {
        "description": "A lung infection acquired outside of a hospital setting.",
        "medicines": ["Amoxicillin", "Azithromycin", "Clarithromycin", "Ceftriaxone", "Levofloxacin"],
        "specialists": ["Pulmonologist", "Infectious Disease Specialist", "General Physician"]
    },
    "Septic Shock": {
        "description": "A severe infection leading to dangerously low blood pressure.",
        "medicines": ["Norepinephrine", "Vancomycin", "Meropenem", "Hydrocortisone", "Dopamine"],
        "specialists": ["Intensivist", "Infectious Disease Specialist", "Emergency Medicine Specialist"]
    },
    "Rheumatoid Arthritis": {
        "description": "An autoimmune disorder causing inflammation in joints.",
        "medicines": ["Methotrexate", "Sulfasalazine", "Hydroxychloroquine", "Adalimumab", "Etanercept"],
        "specialists": ["Rheumatologist", "Orthopedic Specialist", "Physical Therapist"]
    },
    "Congestive Heart Failure": {
        "description": "A chronic condition where the heart doesn't pump blood effectively.",
        "medicines": ["ACE Inhibitors", "Beta Blockers", "Diuretics", "Spironolactone", "Digoxin"],
        "specialists": ["Cardiologist", "General Physician", "Cardiac Surgeon"]
    },
    "Pulmonary Embolism": {
        "description": "A blockage in one of the pulmonary arteries in the lungs.",
        "medicines": ["Heparin", "Warfarin", "Alteplase", "Rivaroxaban", "Dabigatran"],
        "specialists": ["Pulmonologist", "Hematologist", "Emergency Medicine Specialist"]
    },
    "Sepsis": {
        "description": "A life-threatening organ dysfunction caused by a dysregulated immune response to infection.",
        "medicines": ["Vancomycin", "Meropenem", "Piperacillin-Tazobactam", "Cefepime", "Dopamine"],
        "specialists": ["Infectious Disease Specialist", "Intensivist", "Emergency Medicine Specialist"]
    },
    "Liver Cirrhosis": {
        "description": "A late-stage liver disease caused by liver scarring and damage.",
        "medicines": ["Spironolactone", "Furosemide", "Lactulose", "Nadolol", "Rifaximin"],
        "specialists": ["Hepatologist", "Gastroenterologist", "Nutritionist"]
    },
    "Acute Renal Failure": {
        "description": "A sudden loss of kidney function.",
        "medicines": ["Diuretics", "Dopamine", "Calcium Gluconate", "Sodium Bicarbonate", "Epoetin"],
        "specialists": ["Nephrologist", "Critical Care Specialist", "Internal Medicine Specialist"]
    },
    "Urinary Tract Infection": {
        "description": "An infection in any part of the urinary system.",
        "medicines": ["Nitrofurantoin", "Ciprofloxacin", "Amoxicillin-Clavulanate", "Trimethoprim-Sulfamethoxazole", "Cephalexin"],
        "specialists": ["Urologist", "General Physician", "Infectious Disease Specialist"]
    },
    "Hypertension": {
        "description": "A condition in which the force of the blood against the artery walls is too high.",
        "medicines": ["Lisinopril", "Amlodipine", "Losartan", "Hydrochlorothiazide", "Metoprolol"],
        "specialists": ["Cardiologist", "General Physician", "Nephrologist"]
    },
    "Asthma": {
        "description": "A condition in which the airways narrow and swell, causing difficulty in breathing.",
        "medicines": ["Albuterol", "Fluticasone", "Montelukast", "Budesonide", "Salmeterol"],
        "specialists": ["Pulmonologist", "Allergist", "General Physician"]
    },
    "Gastroesophageal Reflux Disease (GERD)": {
        "description": "A digestive disorder where stomach acid irritates the esophagus.",
        "medicines": ["Omeprazole", "Esomeprazole", "Ranitidine", "Lansoprazole", "Pantoprazole"],
        "specialists": ["Gastroenterologist", "General Physician", "Dietitian"]
    }
}

# Function to Clean Text
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Function to Predict Disease
def predict_disease(patient_note, model, tokenizer, label_encoder):
    if not model or not tokenizer or not label_encoder:
        return "Error: Model not loaded properly."

    patient_note = clean_text(patient_note)
    inputs = tokenizer(patient_note, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_label = torch.argmax(logits, dim=1).item()
    predicted_disease = label_encoder.inverse_transform([predicted_label])[0]
    return predicted_disease

# Function to Retrieve Disease Details
def get_disease_details(disease_name):
    return disease_data.get(disease_name, {
        "description": "No details available.",
        "medicines": [],
        "specialists": []
    })

# Streamlit UI
st.title("🩺 Clinical Note Disease Prediction")
st.write("Upload a **medical note (PDF or TXT)**, and this app will predict the **disease** and provide relevant details.")

uploaded_file = st.file_uploader("Upload a clinical note (PDF/TXT)", type=["pdf", "txt"])

if uploaded_file is not None:
    text = ""

    if uploaded_file.name.endswith(".pdf"):
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"❌ Error reading PDF: {e}")
    elif uploaded_file.name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8")

    if text:
        st.subheader("Extracted Text from File:")
        st.write(text[:1000])  # Show a snippet of extracted text

        # Predict disease
        predicted_disease = predict_disease(text, model, tokenizer, label_encoder)
        disease_details = get_disease_details(predicted_disease)

        st.success(f"### Predicted Disease: **{predicted_disease}**")
        st.write(f"**Description:** {disease_details['description']}")

        # Display Medicines
        if disease_details["medicines"]:
            st.write("💊 **Recommended Medicines:**")
            st.write(", ".join(disease_details["medicines"]))

        # Display Specialists
        if disease_details["specialists"]:
            st.write("🩺 **Recommended Specialists:**")
            st.write(", ".join(disease_details["specialists"]))

        # Download extracted text
        st.download_button("Download Extracted Text", text, file_name="extracted_text.txt")

    else:
        st.error("Could not extract text from the file. Please try another file.")

# Footer
st.markdown("---")
st.write("🔬 Developed by **Syed Faizan, MD** | NLP for Clinical Notes | Powered by **BioBERT**")
