try:
    print("📥 Loading data...")

    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import train_test_split
    from preprocess import clean_email
    import joblib

    print("✅ Libraries imported")

    data = {
        "email": [
            "Win a free iPhone now!",
            "Meeting at 10am",
            "Your account is hacked, reset your password",
            "Lunch tomorrow?",
            "Click this link to claim prize"
        ],
        "label": [1, 0, 1, 0, 1]
    }

    print("✅ Data defined")

    df = pd.DataFrame(data)
    print("✅ DataFrame created")

    df['email'] = df['email'].apply(clean_email)
    print("✅ Emails cleaned")

    X_train, X_test, y_train, y_test = train_test_split(df['email'], df['label'], test_size=0.2)
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    print("✅ Model trained and saved")

except Exception as e:
    print("❌ An error occurred:")
    import traceback
    traceback.print_exc()

