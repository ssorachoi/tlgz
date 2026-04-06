# TLGZ - Gen Z Slang Translator

An HCI-focused web application that translates Gen Z slang into standard English.

## 🎯 Features

- **Linear Search Algorithm**: Fast and intuitive slang lookup
- **Sentence Processing**: Detects multiple slang terms in full sentences
- **Machine Learning**: Naive Bayes model predicts meanings for unknown slang
- **Streamlit UI**: Clean, user-friendly interface with immediate feedback
- **Data Cleaning**: Automatic dataset preprocessing and normalization
- **1000+ Slang Terms**: Comprehensive Gen Z vocabulary database

## 📋 Project Structure

```
tlgz_algo/
├── app.py                                          # Main application file
├── genz_dataset_final_augmented (1).csv           # Dataset with slang terms
├── requirements.txt                                # Python dependencies
└── README.md                                       # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📚 How to Use

1. **Type a slang word** - Enter any Gen Z slang term (e.g., "no cap", "slaps", "bussin")
2. **Or enter a full sentence** - TLGZ will detect and translate all slang words
3. **Get instant feedback** - See meanings, examples, and categories
4. **For unknown slang** - AI predicts meanings based on learned patterns

### Example Inputs

- Single term: `no cap`
- Full sentence: `That movie was lowkey fire and the ending was bussin`
- Multiple terms: `That fit is drip no cap fr fr`

## 🏗️ Architecture

### Data Processing
- Loads CSV with columns: `slang, meaning, example, category, language`
- Removes duplicates and handles missing values
- Normalizes text (lowercase, strip whitespace)

### Search Algorithm
- **Linear Search**: O(n) complexity, searches sequentially through dataset
- Simple and effective for understanding algorithm fundamentals

### String Processing
- Tokenizes user input into individual words
- Removes punctuation and normalizes case
- Matches tokens against cleaned dataset

### Machine Learning
- **Algorithm**: Multinomial Naive Bayes
- **Features**: Character n-grams (2-3 character sequences)
- **Training**: 1000+ slang-meaning pairs
- **Prediction**: Estimates meanings for unknown slang

### UI/UX (HCI Principles)
- **Clear Layout**: Organized sections (header, instructions, input, output)
- **Visual Hierarchy**: Color-coded messages and styling
- **User Guidance**: Expandable instructions section
- **Immediate Feedback**: Success/warning/error messages
- **Error Handling**: Graceful degradation with helpful messages
- **Mobile Responsive**: Works on all screen sizes

## 🔧 Technical Details

### Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `pandas` | Data manipulation and analysis |
| `scikit-learn` | Machine learning (Naive Bayes) |
| `numpy` | Numerical computations |

### Key Functions

#### Data Loading & Cleaning
```python
load_and_clean_data()
```
- Loads CSV file
- Removes duplicates
- Handles missing values
- Normalizes text

#### Linear Search
```python
linear_search_slang(dataset, search_term)
```
- O(n) time complexity
- Returns slang info if found

#### Slang Detection
```python
detect_slang_in_sentence(sentence, dataset)
```
- Tokenizes input
- Finds all slang terms in sentence
- Returns list of matches

#### ML Model
```python
train_ml_model(dataset)
predict_meaning(slang_term, vectorizer, model)
```
- Trains Naive Bayes classifier
- Predicts meanings for unknown slang

## 💡 HCI Principles Implemented

1. **User Control & Freedom**
   - Clear button to reset input
   - Expandable instructions
   - Multiple input options

2. **Visibility of System Status**
   - Success messages (✅)
   - Warning messages (⚠️)
   - Error messages (❌)

3. **Match Between System & Real World**
   - Familiar icons and language
   - Gen Z appropriate terminology
   - Examples show real usage

4. **Aesthetic & Minimalist Design**
   - Clean layout
   - Proper spacing and typography
   - Color-coded feedback
   - Centered, organized content

5. **Error Prevention & Recovery**
   - Handles missing data gracefully
   - Tries ML prediction if lookup fails
   - Clear error messages with suggestions

## 🔄 Data Pipeline

```
CSV File (genz_dataset_final_augmented (1).csv)
         ↓
    [Data Cleaning]
    - Remove duplicates
    - Handle nulls
    - Normalize text
         ↓
    [Cleaned Dataset]
    - 800+ unique slang terms
    - Ready for search & ML
         ↓
    [User Input] → [Linear Search] → [Found!] → [Display Result]
                                  ↘         ↙
                               [Not Found]
                                  ↓
                          [ML Prediction]
                                  ↓
                              [Display AI Result]
```

## 📊 Dataset Information

The dataset includes:
- **slang_term**: The Gen Z slang word
- **meaning**: Standard English translation
- **slang_sentence**: Example usage of the slang
- **normal_sentence**: Example translation
- **category**: Type of slang (expression, abbreviation, etc.)
- **semantic_category**: Linguistic category
- **frequency_score**: Popularity score

## 🔍 Troubleshooting

### Issue: "Dataset file not found"
- Ensure `genz_dataset_final_augmented (1).csv` is in the same directory as `app.py`
- Check file name spelling (including spaces and parentheses)

### Issue: Module not found error
- Run: `pip install -r requirements.txt`
- Or individually: `pip install streamlit pandas scikit-learn numpy`

### Issue: Slow performance
- First run trains ML model (cached afterwards)
- Subsequent runs are much faster
- Clear browser cache if slow

## 📱 Browser Compatibility

Works on:
- Chrome / Chromium (recommended)
- Firefox
- Safari
- Edge
- Mobile browsers

## 🎓 Learning Outcomes

This project teaches:
- Linear search algorithm
- String tokenization and processing
- Machine learning classification
- Data cleaning and preprocessing
- Web application development with Streamlit
- HCI principles and user experience design
- Feature engineering for text data

## 📝 Comments & Code Quality

The code includes:
- Detailed docstrings for all functions
- Inline comments explaining complex logic
- Type hints and clear variable names
- Structured sections with clear separation
- Beginner-friendly explanations

## 🚀 Deployment Options

### Local
```bash
streamlit run app.py
```

### Streamlit Cloud
1. Push code to GitHub
2. Go to share.streamlit.io
3. Connect your repository
4. Deploy with one click

### Docker
```bash
docker run -p 8501:8501 -v $(pwd):/app streamlit/streamlit-demo
```

## 📄 License & Credits

**Created for**: HCI-focused application development
**Focus**: User-centered design, accessibility, and clear feedback

---

**Ready to translate? Run `streamlit run app.py` and start exploring Gen Z slang! 🧢**
