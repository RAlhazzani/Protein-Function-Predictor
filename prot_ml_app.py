import streamlit as st
import pandas as pd
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# Part 1: Bioinformatics Feature Extraction
# This section converts amino acid sequences into numerical data.
# ---------------------------------------------------------
def get_protein_features(sequence):
    """
    Extracts physicochemical properties from a protein sequence.
    """
    try:
        # Ensure the sequence is in uppercase
        seq = sequence.upper()
        
        # Use BioPython for protein analysis
        analysed_seq = ProteinAnalysis(seq)
        
        # Extract features (These act as input features for the model)
        features = {
            'Molecular_Weight': analysed_seq.molecular_weight(),
            'Aromaticity': analysed_seq.aromaticity(),
            'Instability_Index': analysed_seq.instability_index(),
            'Isoelectric_Point': analysed_seq.isoelectric_point(),
            'Helix_Fraction': analysed_seq.secondary_structure_fraction()[0]
        }
        return features
    except Exception as e:
        return None

# ---------------------------------------------------------
# Part 2: Data Preparation & Model Training (Machine Learning)
# In a real-world scenario, this would load a CSV dataset.
# ---------------------------------------------------------
@st.cache_resource  # Caches the model to improve app performance
def train_model():
    # 1. Generate synthetic data for demonstration purposes
    # Classifying proteins into: 'Soluble' vs 'Insoluble'
    data = []
    labels = []
    
    # Generate 100 random synthetic protein sequences
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    for _ in range(100):
        # Create a random sequence
        seq_len = np.random.randint(50, 200)
        seq = "".join(np.random.choice(list(amino_acids), seq_len))
        
        # Extract features
        feats = get_protein_features(seq)
        
        # Define a simple synthetic rule for classification:
        # If High Molecular Weight & High Isoelectric Point -> Insoluble
        # Else -> Soluble
        if feats['Molecular_Weight'] > 100 and feats['Isoelectric_Point'] > 6:
            label = "Insoluble"
        else:
            label = "Soluble"
            
        data.append(list(feats.values()))
        labels.append(label)

    # 2. Convert data to a DataFrame
    df = pd.DataFrame(data, columns=['Molecular_Weight', 'Aromaticity', 'Instability_Index', 'Isoelectric_Point', 'Helix_Fraction'])
    
    # 3. Train the Model
    X = df  # Features
    y = labels  # Target variable
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Calculate Accuracy
    acc = accuracy_score(y_test, model.predict(X_test))
    
    return model, acc

# ---------------------------------------------------------
# Part 3: Web Application Interface (Streamlit)
# ---------------------------------------------------------

# Page Configuration
st.set_page_config(page_title="Protein Classifier AI", page_icon="🧬")

# Title and Description
st.title("🧬 AI Protein Function Predictor")
st.markdown("""
This application leverages **Machine Learning** to predict protein solubility properties 
based on physicochemical characteristics extracted from the amino acid sequence.
""")

# Sidebar: Model Information
st.sidebar.header("Model Performance")
model, accuracy = train_model()
st.sidebar.success(f"Model Accuracy: {accuracy * 100:.1f}%")
st.sidebar.info("Algorithm: Random Forest Classifier")

# Input Section
input_sequence = st.text_area("Enter Amino Acid Sequence:", height=150, placeholder="Example: MVLSPADKTNVKAAWGKVGAHAGEYGAE...")

# Prediction Button
if st.button("Analyze & Predict 🚀"):
    if input_sequence:
        # 1. Extract Features
        features = get_protein_features(input_sequence)
        
        if features:
            # Display Extracted Features
            st.subheader("1. Extracted Bio-Features:")
            features_df = pd.DataFrame([features])
            st.table(features_df)
            
            # 2. Make Prediction
            prediction = model.predict(features_df)[0]
            probability = model.predict_proba(features_df).max()
            
            # Display Result
            st.subheader("2. AI Prediction Result:")
            
            # Color-coded output
            if "Soluble" == prediction:
                st.success(f"Prediction: **{prediction}**")
            else:
                st.warning(f"Prediction: **{prediction}**")
                
            st.write(f"Confidence Score: **{probability*100:.2f}%**")
            
        else:
            st.error("Error: The sequence contains invalid characters. Please use standard amino acid codes.")
    else:
        st.warning("Please enter a protein sequence first.")

# Footer
st.markdown("---")
st.caption("Developed by Raneem Alhazzani | ra.alhazzani@outlook.com")