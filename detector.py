import joblib
from preprocess import clean_email

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Ask for email content
email = input("📩 Enter email content:\n> ")

# Clean and vectorize
cleaned = clean_email(email)
vectorized = vectorizer.transform([cleaned])

# Predict
prediction = model.predict(vectorized)[0]

# Show result
if prediction == 1:
    print("⚠️ This email is likely MALICIOUS.")
else:
    print("✅ This email seems safe.")
