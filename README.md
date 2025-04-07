# 🐦 TweetLike — Who Said That?

A fun pop culture game where users guess which celebrity tweeted a quote — with a twist: some tweets are AI-generated. Includes an ML model that tells you which celeb *you* tweet like.
## Important
You might need to recreate similar folder structure as was initially, there was an issue uploading / creating folders here, the structure is following:

<img width="326" alt="Screenshot 2025-04-06 at 6 24 14 PM" src="https://github.com/user-attachments/assets/47a38793-2938-432e-9f57-1e1e42251c38" />

You can ignore avatars folder, as it wasn't used in final version.

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

