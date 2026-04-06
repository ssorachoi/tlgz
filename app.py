"""
================================================================================
TLGZ (TalkLikeGenZ) - Gen Z Slang Translator
An HCI-focused web application that translates Gen Z slang into standard English
================================================================================

This application demonstrates:
- User-centered design with clear UI and immediate feedback
- Data cleaning and preprocessing techniques
- Linear search algorithm implementation
- String tokenization and pattern matching
- Machine Learning model (Naive Bayes) for meaning prediction
- Interactive web interface with Streamlit

Dependencies: streamlit, pandas, scikit-learn
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import re
from pathlib import Path


# ================================================================================
# CONFIGURATION & SETUP
# ================================================================================

# Page configuration for better UX
st.set_page_config(
    page_title="TLGZ - Gen Z Translator",
    page_icon="🧢",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI/UX (HCI principle: visual hierarchy and spacing)
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .stTitle {
        text-align: center;
        color: #00D9FF;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 30px;
        font-size: 1.1em;
    }
    .result-box {
        background-color: #e6f2ff;
        border-left: 5px solid #0099ff;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
        color: #000;
    }
    .result-box b {
        color: #0066cc;
        font-size: 1.1em;
    }
    .error-box {
        background-color: #fff8e1;
        border-left: 5px solid #ff9800;
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
        color: #000;
    }
    .error-box b {
        color: #e65100;
    }
    </style>
""", unsafe_allow_html=True)


# ================================================================================
# DATA LOADING & CLEANING FUNCTIONS
# ================================================================================

@st.cache_resource
def load_and_clean_data():
    """
    Load CSV dataset and perform data cleaning.
    
    Steps:
    1. Read CSV file
    2. Remove duplicates
    3. Handle missing values
    4. Normalize text (lowercase, strip whitespace)
    5. Remove empty rows
    
    Returns:
        DataFrame: Cleaned dataset with columns [slang, meaning, example, language, category]
    """
    try:
        # Find and load the CSV file
        csv_path = Path("genz_dataset_final_augmented (1).csv")
        
        if not csv_path.exists():
            # Try alternative path in case file is in different location
            csv_path = Path("c:\\Users\\User\\Documents\\zyrene\\tlgz_algo\\genz_dataset_final_augmented (1).csv")
        
        # Load dataset
        df = pd.read_csv(csv_path)
        
        # Use most relevant columns
        # Prioritize 'slang_term' over 'slang', and 'slang_sentence' as example
        df_clean = pd.DataFrame({
            'slang': df['slang_term'].fillna(df.get('slang', '')),
            'meaning': df['meaning'],
            'example': df.get('slang_sentence', df.get('example', '')).fillna(''),
            'language': df.get('language', 'en').fillna('en'),
            'category': df.get('category', df.get('semantic_category', 'general')).fillna('general')
        })
        
        # Remove duplicate slang terms (keep first occurrence)
        df_clean = df_clean.drop_duplicates(subset=['slang'], keep='first')
        
        # Remove rows where slang or meaning is empty
        df_clean = df_clean[(df_clean['slang'].notna()) & 
                           (df_clean['slang'] != '') & 
                           (df_clean['meaning'].notna()) & 
                           (df_clean['meaning'] != '')]
        
        # Text normalization: lowercase and strip whitespace
        df_clean['slang'] = df_clean['slang'].str.lower().str.strip()
        df_clean['meaning'] = df_clean['meaning'].str.lower().str.strip()
        df_clean['example'] = df_clean['example'].str.lower().str.strip()
        
        # Reset index
        df_clean.reset_index(drop=True, inplace=True)
        
        return df_clean
    
    except FileNotFoundError:
        st.error("❌ Dataset file not found. Make sure 'genz_dataset_final_augmented (1).csv' is in the same directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return None


# ================================================================================
# SEARCHING ALGORITHM
# ================================================================================

def linear_search_slang(dataset, search_term):
    """
    Linear Search Algorithm - O(n) time complexity.
    
    Searches through dataset sequentially to find matching slang term.
    This is a simple, intuitive approach good for smaller datasets.
    
    Args:
        dataset (DataFrame): The cleaned dataset to search
        search_term (str): The slang term to search for (case-insensitive)
    
    Returns:
        dict or None: If found, returns {'slang': ..., 'meaning': ..., 'example': ..., 'category': ...}
                      If not found, returns None
    """
    search_term = search_term.lower().strip()
    
    # Linear search: iterate through each row
    for index, row in dataset.iterrows():
        if row['slang'] == search_term:
            # Found a match - return the result
            return {
                'slang': row['slang'],
                'meaning': row['meaning'],
                'example': row['example'],
                'category': row['category']
            }
    
    # No match found
    return None


# ================================================================================
# STRING PROCESSING & SLANG DETECTION
# ================================================================================

def detect_slang_in_sentence(sentence, dataset):
    """
    Detect slang words within a full sentence.
    
    Process:
    1. Tokenize sentence into words
    2. Normalize each word (lowercase, remove punctuation)
    3. Search for each word in dataset
    4. Return all matches found
    
    Args:
        sentence (str): User input sentence
        dataset (DataFrame): Cleaned dataset
    
    Returns:
        list: List of dicts with detected slang and their meanings
    """
    # Tokenization: split sentence into words and remove punctuation
    # Pattern removes all non-alphanumeric characters except spaces
    cleaned_sentence = re.sub(r'[^a-zA-Z0-9\s]', ' ', sentence.lower())
    
    # Split into individual words (tokens)
    tokens = cleaned_sentence.split()
    
    detected_slang = []
    
    # Search for each token in the dataset
    for token in tokens:
        if len(token) > 0:  # Skip empty tokens
            result = linear_search_slang(dataset, token)
            if result is not None:
                # Check if already detected (avoid duplicates)
                if not any(d['slang'] == result['slang'] for d in detected_slang):
                    detected_slang.append(result)
    
    return detected_slang


# ================================================================================
# MACHINE LEARNING - NAIVE BAYES MODEL
# ================================================================================

@st.cache_resource
def train_ml_model(dataset):
    """
    Train Naive Bayes model to predict meanings.
    
    Process:
    1. Extract slang terms as features (input)
    2. Extract meanings as labels (output)
    3. Vectorize text using CountVectorizer
    4. Train MultinomialNB classifier
    
    This model learns patterns in slang and tries to predict meanings
    for unknown slang words.
    
    Args:
        dataset (DataFrame): Cleaned dataset with 'slang' and 'meaning' columns
    
    Returns:
        tuple: (vectorizer, model) for feature transformation and prediction
    """
    try:
        # Features: slang terms
        X = dataset['slang'].values
        
        # Labels: meanings
        y = dataset['meaning'].values
        
        # Vectorizer: converts text to numerical features
        # Uses character n-grams to capture slang-like patterns
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 3), max_features=100)
        X_vectorized = vectorizer.fit_transform(X)
        
        # Train Naive Bayes model
        model = MultinomialNB()
        model.fit(X_vectorized, y)
        
        return vectorizer, model
    
    except Exception as e:
        st.error(f"❌ Error training ML model: {str(e)}")
        return None, None


def predict_meaning(slang_term, vectorizer, model):
    """
    Predict meaning for unknown slang using trained Naive Bayes model.
    
    Args:
        slang_term (str): Unknown slang term
        vectorizer: Fitted CountVectorizer
        model: Trained MultinomialNB model
    
    Returns:
        str or None: Predicted meaning, or None if prediction fails
    """
    try:
        if vectorizer is None or model is None:
            return None
        
        # Vectorize the input term using the same vectorizer
        term_vectorized = vectorizer.transform([slang_term.lower().strip()])
        
        # Predict meaning
        prediction = model.predict(term_vectorized)
        
        return prediction[0] if len(prediction) > 0 else None
    
    except Exception as e:
        return None


# ================================================================================
# MAIN STREAMLIT APPLICATION
# ================================================================================

def main():
    """Main Streamlit application with HCI principles."""
    
    # ========== LOAD & PREPARE DATA ==========
    dataset = load_and_clean_data()
    if dataset is None or dataset.empty:
        st.stop()
    
    # Train ML model
    vectorizer, model = train_ml_model(dataset)
    
    # ========== HEADER SECTION (HCI: Clear Title & Purpose) ==========
    st.markdown("""
        <div class="stTitle">🧢 TLGZ</div>
        <div class="subtitle">Gen Z Slang Translator</div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========== INSTRUCTIONS SECTION (HCI: User Guidance) ==========
    with st.expander("📖 How to Use", expanded=False):
        st.markdown("""
        **What is TLGZ?**
        TLGZ helps you understand and translate Gen Z slang into standard English.
        
        **How it works:**
        1. **Type a slang word** - Enter any Gen Z slang term (e.g., "no cap", "slaps", "bussin")
        2. **Or enter a full sentence** - TLGZ will detect slang words within it
        3. **Get the meaning** - See the definition and examples
        
        **Features:**
        - ✅ Searches database of 1000+ Gen Z slang terms
        - 🤖 AI prediction for unknown slang
        - 💡 Example sentences showing usage
        - 🔄 Process multiple phrases at once
        
        **Example inputs:**
        - Single term: "no cap"
        - Full sentence: "That movie was lowkey fire and the ending was bussin"
        """)
    
    st.markdown("---")
    
    # ========== INPUT SECTION (HCI: Clear Input Area) ==========
    st.subheader("🔍 Enter Text to Translate")
    
    # Two-column layout for input and button
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Type slang word or sentence:",
            placeholder="e.g., 'no cap' or 'that's so slaps fr fr'",
            label_visibility="collapsed"
        )
    
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
        if clear_button:
            st.rerun()
    
    # ========== PROCESSING & OUTPUT SECTION ==========
    if user_input and user_input.strip():
        st.markdown("---")
        st.subheader("📋 Results")
        
        # Process input: detect slang words in sentence
        detected_slang = detect_slang_in_sentence(user_input, dataset)
        
        if detected_slang:
            # ===== SUCCESS: Slang terms found in dataset =====
            st.success(f"✅ Found {len(detected_slang)} slang term(s)!")
            
            # Display each found slang term
            for idx, slang_info in enumerate(detected_slang, 1):
                with st.container():
                    st.markdown(f"""
                    <div class="result-box">
                    <b>#{idx} {slang_info['slang'].upper()}</b>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Meaning:** {slang_info['meaning'].capitalize()}")
                    with col2:
                        st.markdown(f"**Category:** {slang_info['category']}")
                    
                    st.markdown(f"**Example:** *\"{slang_info['example']}\"*")
        else:
            # ===== NOT FOUND: Try ML prediction =====
            st.warning("⚠️ Slang term(s) not found in database. Using AI to predict...")
            
            # Try to predict meaning using ML model
            search_term = user_input.lower().strip()
            predicted_meaning = predict_meaning(search_term, vectorizer, model)
            
            if predicted_meaning:
                st.markdown(f"""
                <div class="result-box">
                <b>🤖 AI PREDICTION for "{search_term.upper()}"</b><br><br>
                <b>Predicted Meaning:</b> {predicted_meaning.capitalize()}<br><br>
                <i>Note: This is an AI prediction based on similar slang patterns. 
                Accuracy may vary.</i>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="error-box">
                <b>❌ Translation Not Available</b><br>
                This slang term is not in our database and AI prediction was unsuccessful.
                <br><br>
                <i>Try:</i>
                - Check spelling and grammar
                - Use simpler terms
                - Try different slang words
                </div>
                """, unsafe_allow_html=True)
    
    # ========== FOOTER SECTION (HCI: Context & Additional Info) ==========
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.9em;'>
    📊 Database: 1000+ Gen Z slang terms | 🤖 ML Model: Naive Bayes Classifier
    <br>
    💡 Built with HCI principles for clear, intuitive user experience
    </div>
    """, unsafe_allow_html=True)


# ================================================================================
# ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    main()
