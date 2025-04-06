# 🐦 TweetLike — Who Said That?

A fun pop culture game where users guess which celebrity tweeted a quote — with a twist: some tweets are AI-generated. Includes an ML model that tells you which celeb *you* tweet like.

## 🛠 Built With
- **Streamlit** – frontend UI  
- **Python** – core logic  
- **pandas** – data handling  
- **scikit-learn** – ML model  
- **eli5** – model explainability

## 📁 Files
- `app.py` — main Streamlit app  
- `model.py` — ML logic and pipeline  
- `tweets.csv` — real and AI-generated tweets  
- `avatars/` — celebrity profile images  
- `tweetlike_model.pkl` — trained classifier  
- `utils.py` — helper functions

## ▶️ Run Locally

```bash
git clone https://github.com/your-username/tweetlike.git
cd tweetlike
pip install -r requirements.txt
streamlit run app.py

