import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Improved dataset
data = {
    "text": [
        "NASA announces water discovery on Mars.",
        "Aliens found on Earth, scientists confirm.",
        "President signs new education bill.",
        "COVID-19 cured with orange juice, reports say.",
        "Apple releases new iPhone with satellite features.",
        "Bill Gates owns the sun now, claims viral post.",
        "Doctors recommend vaccine to prevent flu.",
        "World will end in 2026, YouTuber warns.",
        "Scientists discover ancient city beneath the Sahara.",
        "Drinking bleach can cure any disease, Facebook post claims.",
        "Stock markets hit record high amid tech boom.",
        "Man grows wings after eating GMO corn, experts say.",
        "University opens AI-powered smart campus.",
        "Wearing a hat boosts intelligence, TikTok users suggest."
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Real, 0 = Fake
}

df = pd.DataFrame(data)

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

model = LogisticRegression()
model.fit(X, y)

# Save model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model trained and saved as model.pkl and vectorizer.pkl")
