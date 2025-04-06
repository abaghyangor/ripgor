import streamlit as st
import pandas as pd
import numpy as np
from model import get_model
import joblib
import eli5
import streamlit.components.v1 as components


if 'fake_tweet_count' not in st.session_state:
    st.session_state.fake_tweet_count = 0
    st.session_state.fake_tweets_list = []
    st.session_state.total_questions = 0


# Function to set background color to light yellow and text to black
def set_light_theme():
    st.markdown(
        """
        <style>
        /* Set background color to light yellow and text to black */
        .stApp {
            background-color: #15202b;
            font-family: 'Helvetica Neue', sans-serif;
            color: white;
        }

        /* Header style - White text */
        h1, h2, h3, h4, .css-1q2bbj3 {
            color: white;
        }

        /* Buttons styled with a light color */
        .stButton>button {
            background-color: #2E5984;
            color: white;
            border-radius: 25px;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            box-shadow: none;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #FFD700;
        }
        /* ✅ Custom style for 'Next question' button only */
        button[data-testid="next_question"] {
            background-color: #d4edda !important; /* Light green */
            color: black !important;
        }

        button[data-testid="next_question"]:hover {
            background-color: #c3e6cb !important; /* Slightly darker on hover */
        }

        /* Main content area without sidebar */
        .main-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            width: 100%;
            max-width: 900px;
            margin: 0 auto;
            padding-top: 50px;
            text-align: center;
        }

        /* Quit button in the top-right corner */
        .quit-btn {
            position: absolute;
            top: 20px;
            right: 20px;
            background-color: #FFDD44;
            color: black;
            padding: 10px 20px;
            border-radius: 20px;
            cursor: pointer;
            border: none;
            font-size: 16px;
        }

        .quit-btn:hover {
            background-color: #FFD700;
        }
        /* Set global default text color to black */
        html, body, [class*="css"] {
        color: black !important;
        }

        /* Make all Streamlit alert box text black */
        div.stAlert {
        color: black !important;
        }
        /* set start game button color */
        button[data-testid="start_game"] {
            background-color: #90EE90 !important; /* Light yellow */
            color: black !important;
        }
        button[data-testid="start_game"]:hover {
            background-color: #98FB98 !important; /* Slightly darker on hover */
        }
        /* Match buttons inside columns */
        div[data-testid="column"] button {
            width: 100% !important;
            padding: 20px 0 !important;
            font-size: 24px !important;
            border-radius: 28px !important;
        }
        </style>
        """, unsafe_allow_html=True
    )

# HOME PAGE FUNCTION
def Home():
    st.set_page_config(page_title="WHO SAID THAT?", layout="wide")
    
    # Set light theme style (light yellow background, black text)
    set_light_theme()

    # Title and Content Container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>🎮 WHO SAID THAT? 🎤", unsafe_allow_html=True)

    
    st.markdown("""
    **Welcome to "WHO SAID THAT?"** 🤩  
    When we think of our favorite celebrities, we all can tie a quote to them. Sometimes, we will read a quote and think, 'I know who said that!'.  
    This game tests your skills in being able to tell which celebrity tweeted what.
    The game consists of **10 questions**, and you have to guess the author of each tweet.
    """)

    st.markdown("""You can choose between two game modes:
    - **Easy mode**: You will pick the answer based on options.
    - **Hard mode**: You will have to guess the author without any options.
    """)

    st.write("#### Let's get started!")

    if 'game_started' not in st.session_state:
        st.session_state.game_started = False
    if 'game_mode' not in st.session_state:
        st.session_state.game_mode = None
    
    ### BUTTONS ###
    st.write('Choose your game mode:')
    if st.button("Easy mode", key="easy_mode"):
        reset_game_state()
        st.session_state.game_mode = "easy"

    if st.button("Hard mode", key="hard_mode"):
        reset_game_state()
        st.session_state.game_mode = "hard"

    if st.session_state.game_mode:
        st.write(f"You have chosen the **{st.session_state.game_mode.capitalize()}** mode.")
        if st.button("Start game", key="start_game"):
            st.session_state.page = "GameStarted"
            st.rerun()
    
    # ML Model & Prediction
    st.markdown("""
        <style>
        /* Works across Streamlit versions/themes */
        label[data-testid="stCheckbox-label"] {
            color: black !important;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    model = joblib.load('tweetlike_model.pkl')

    st.markdown("<h1 style='text-align: center;'> 🎤 Who Do You Tweet Like 💅", unsafe_allow_html = True)
    user_tweet = st.text_area("Write your own tweet:")

    st.markdown("""
    <style>
    label[data-testid="stCheckbox-label"] {
        font-weight: bold;
        color: black !important;  /* or white for dark themes */
    }
    </style>
""", unsafe_allow_html=True)
    
    simple_explanation = st.checkbox('### Show Simple Explanation')
    detailed_explanation = st.checkbox('### Show Detailed Explanation')

    if st.button("Guess"):
        if user_tweet.strip() == "":
            st.warning("### Please write something first!")
        else:
            pred = model.predict([user_tweet])[0]
            probs = model.predict_proba([user_tweet])[0]
            top3 = sorted(zip(model.classes_, probs), key=lambda x: x[1], reverse=True)[:3]

            st.success(f"## You sound like **{pred}**!")
            st.markdown("### Top 3 Predictions:")
            for celeb, prob in top3:
                st.markdown(f"- **{celeb}**: {prob:.1%}")

            # Explanation logic starts here
            st.markdown("### Why we guessed that:")

            if detailed_explanation:
                vectorizer = model.named_steps['tfidf']
                classifier = model.named_steps['clf']
                X_transformed = vectorizer.transform([user_tweet])

                explanation = eli5.explain_prediction(
                    classifier,
                    X_transformed[0],
                    feature_names=vectorizer.get_feature_names_out()
                )
                html = eli5.format_as_html(explanation)
                custom_html = f"""
<div style="background-color: white; color: black; padding: 20px; border-radius: 12px;">
{html}
</div>
"""
                components.html(custom_html, height=400, scrolling=True)

            if simple_explanation:
                vectorizer = model.named_steps['tfidf']
                top_words = sorted(
                    user_tweet.lower().split(),
                    key=lambda word: word in vectorizer.vocabulary_,
                    reverse=True
                )[:5]
                st.markdown("Your tweet had words like:")
                st.markdown(", ".join([f"`{w}`" for w in top_words]))

            

        st.markdown('</div>', unsafe_allow_html=True)


### MANAGING DATA ###
# Load the dataset
def load_data(file_path):

    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Get a random tweet and its author
def get_random_tweet(df):
    random_row = df.sample(n=1).iloc[0]
    tweet = random_row['tweet']
    author = random_row['author']
    return tweet, author

def get_options(df, correct_author):
    # Filter out the correct author first
    other_authors = df[df['author'] != correct_author]['author'].unique()
    sampled_authors = np.random.choice(other_authors, size=3, replace=False).tolist()
    sampled_authors.append(correct_author)
    np.random.shuffle(sampled_authors)
    return sampled_authors

def generate_question_easy(df):
    tweet, correct_author = get_random_tweet(df)
    options = get_options(df, correct_author)
    return correct_author, tweet, options

def generate_question_hard(df):
    tweet, correct_author = get_random_tweet(df)
    return correct_author, tweet
def display_tweet(tweet, author = None, avatar = None):
    # fallback values
    display_name = author if author else "Unknown Author"
    display_avatar = avatar if avatar else "https://abs.twimg.com/sticky/default_profile_images/default_profile_400x400.png"
    
    st.markdown(
        f"""
        <div style="border: 1px solid #2f3336; background-color: #15202b; border-radius: 12px;
            padding: 15px; margin: 20px auto; max-width: 550px;
            box-shadow: 0px 2px 5px rgba(0,0,0,0.4);
            font-family: 'Segoe UI', sans-serif;">
            <div style="display: flex; align-items: center;">
                <img src="{display_avatar}" width="48" height="48" style="border-radius: 50%; margin-right: 10px;">
                <div>
                    <span style="font-weight: 600; font-size: 16px; color: white;">{display_name}</span>
                    <img src="https://upload.wikimedia.org/wikipedia/commons/e/e4/Twitter_Verified_Badge.svg" width="16" height="16" style="margin-left: 4px; vertical-align: text-bottom;">
                    <br>
                    <span style="color: #8899a6;">@{author.lower().replace(' ', '') if author else 'Unknown Author'} · Apr 5, 2025</span>
                </div>
            </div>
            <div style="margin-top: 12px; font-size: 18px; line-height: 1.5; color: white;">
                {tweet}
            </div>
            <div style="margin-top: 14px; display: flex; justify-content: space-around; color: #8899a6; font-size: 14px;">
                <div><img src="https://img.icons8.com/ios-glyphs/20/8899a6/speech-bubble.png"/> 17</div>
                <div><img src="https://img.icons8.com/ios-glyphs/20/8899a6/share.png"/> 112</div>
                <div><img src="https://img.icons8.com/ios-glyphs/20/8899a6/like.png"/> 683</div>
                <div><img src="https://img.icons8.com/ios-glyphs/20/8899a6/view-file.png"/> 13.2K</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


#reset the game state
def reset_game_state():
    for key in ['questions', 'current_q', 'answered', 'selected_option', 'score', 'hard_answer']:
        if key in st.session_state:
            del st.session_state[key]

def easy_question(df):
    if 'questions' not in st.session_state:
        st.session_state.questions =[generate_question_easy(df) for _ in range(10)]
        st.session_state.current_q=0
        st.session_state.answered=False
        st.session_state.selected_option=None
        st.session_state.score=0
    q_idx= st.session_state.current_q


    # Assuming tweet is a row from your DataFrame
    correct_author, tweet_text, options = st.session_state.questions[q_idx]

    # Find the row for this tweet
    row = df[df["tweet"] == tweet_text].iloc[0]
    is_real = row["is_real"]

    # Track AI tweet info
    st.session_state.total_questions += 1
    if not is_real and tweet_text not in st.session_state.fake_tweets_list:
        st.session_state.fake_tweet_count += 1
        st.session_state.fake_tweets_list.append(tweet_text)

    #score display
    q_idx = st.session_state.current_q
    score = st.session_state.score
    progress = (q_idx+1) / 10

# Display progress bar and score tracker side by side
    col1, col2 = st.columns([4, 1])
    with col1:
        st.progress(progress, text=f"Question {q_idx + 1} of 10")
    with col2:
        st.markdown(
            f"""
            <div style="background-color: #ffffff; color: black; padding: 8px 12px;
                        border-radius: 12px; text-align: center; font-weight: bold;
                        box-shadow: 1px 1px 5px rgba(0,0,0,0.1); margin-top: 6px;">
                Score: {score}
            </div>
            """, unsafe_allow_html=True
        )

    #display the tweet
    correct_author, tweet, options = st.session_state.questions[q_idx]
    # render the options as buttons
    if not st.session_state.answered:
        display_tweet(tweet)
        cols = st.columns([0.6]*len(options))
        for i, option in enumerate(options):
            with cols[i]:
                if st.button(option, key=f"option_{q_idx}_{i}"):
                    st.session_state.selected_option = option
                    st.session_state.answered = True
                    if option == correct_author:
                        st.session_state.score += 1
                    st.rerun()

    # if the question has been answered, display the correct answer and the selected answer
    else:
        display_tweet(tweet, author = row['author'])
        cols = st.columns(len(options))
        for i, option in enumerate(options):
            with cols[i]:
                if option == correct_author:
                    bg = "#d4edda"  # green
                    icon = "✅"
                elif option == st.session_state.selected_option:
                    bg = "#f8d7da"  # red
                    icon = "❌"
                else:
                    bg = "#e2e3e5"  # gray
                    icon = ""

                st.markdown(
                    f"""
                    <div style='background-color: {bg}; color: black; padding: 10px 16px; border-radius: 20px;
                                text-align: center; font-weight: bold; box-shadow: 1px 1px 3px rgba(0,0,0,0.1);'>
                        {icon} {option}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        
        if q_idx <9:
            st.markdown('<div data-testid="next_question">', unsafe_allow_html=True)
            if st.button("Next question", key="next_question"):
                st.session_state.current_q += 1
                st.session_state.answered = False
                st.session_state.selected_option = None
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            if st.session_state.total_questions > 0:
                percent_fake = (st.session_state.fake_tweet_count / 10) * 100

                st.markdown(f"### 🤖 AI-Generated Tweet Summary")
                st.markdown(f"🧠 **{percent_fake:.1f}%** of the tweets you saw were AI-generated.")

                if st.session_state.fake_tweets_list:
                    st.markdown("Here are the AI-generated tweets you saw:")
                    for fake_tweet in st.session_state.fake_tweets_list:
                        st.markdown(f"> {fake_tweet}")

            st.markdown("### 🏁 Final Results")
            st.balloons()
            score = st.session_state.score
            if score == 10:
                st.markdown(
    f"<div style='background-color: #fff3cd; color: black; padding: 12px; border-radius: 8px;'>  🎉 You scored {score}/10. WOW! </div>",
    unsafe_allow_html=True
)

            elif score >= 7:
                st.markdown(
    f"<div style='background-color: #d4edda; color: black; padding: 12px; border-radius: 8px;'>👏 Great job! You scored {score}/10!</div>",
    unsafe_allow_html=True
)
            elif score >= 4:
                st.markdown(
    f"<div style='background-color: #d1ecf1; color: black; padding: 12px; border-radius: 8px;'>ℹ️ You scored {score}/10. Not bad — give it another go!</div>",
    unsafe_allow_html=True
)

            else:
                st.markdown(
    f"<div style='background-color: #fff3cd; color: black; padding: 12px; border-radius: 8px;'>⚠️ You scored {score}/10. Tough round — try again! 💪</div>",
    unsafe_allow_html=True
)
            if st.button("🔁 Play again", key="play_again"):
                reset_game_state()
                st.session_state.page = "Home"
                st.rerun()

def hard_question(df):
    if 'questions' not in st.session_state:
        st.session_state.questions =[generate_question_hard(df) for _ in range(10)]
        st.session_state.current_q=0
        st.session_state.answered=False
        st.session_state.selected_option=None
        st.session_state.score=0
    q_idx= st.session_state.current_q
    correct_author, tweet = st.session_state.questions[q_idx]
    st.markdown(f"**Tweet {q_idx+1}**:")

    # Getting the tweet and is_real info

    correct_author, tweet_text = st.session_state.questions[q_idx]
    row = df[df["tweet"] == tweet_text].iloc[0]
    is_real = row["is_real"]

    # Track AI tweet info
    st.session_state.total_questions += 1
    if not is_real and tweet_text not in st.session_state.fake_tweets_list:
        st.session_state.fake_tweet_count += 1
        st.session_state.fake_tweets_list.append(tweet_text)

    #score display
    q_idx = st.session_state.current_q
    score = st.session_state.score
    progress = (q_idx+1) / 10

    # Display progress bar and score tracker side by side
    col1, col2 = st.columns([4, 1])
    with col1:
        st.progress(progress, text=f"Question {q_idx + 1} of 10")
    with col2:
        st.markdown(
            f"""
            <div style="background-color: #ffffff; color: black; padding: 8px 12px;
                        border-radius: 12px; text-align: center; font-weight: bold;
                        box-shadow: 1px 1px 5px rgba(0,0,0,0.1); margin-top: 6px;">
                Score: {score}
            </div>
            """, unsafe_allow_html=True
        )
    #display the tweet
    if not st.session_state.answered:
        display_tweet(tweet)
        st.text_input("Type your answer here:", key="hard_answer")
        if st.button("Submit", key="submit_answer"):
            st.session_state.answered = True
            st.session_state.selected_option = st.session_state.hard_answer
            if st.session_state.selected_option.lower() == correct_author.lower():
                st.session_state.score +=1
            st.rerun()
    else:
        display_tweet(tweet, author=row['author'])
        # Determine styling
        if st.session_state.selected_option.lower() == correct_author.lower():
            bg = "#d4edda"
            icon = "✅"
        else:
            bg = "#f8d7da"
            icon = "❌"
        st.markdown(
            f"""
            <div style='background-color: {bg}; color: black; padding: 10px 16px; border-radius: 20px;
                margin-bottom: 10px; font-weight: bold; box-shadow: 1px 1px 3px rgba(0,0,0,0.1);'>
                {icon} {st.session_state.selected_option} The correct answer was: {correct_author}
            </div>
            """,
            unsafe_allow_html=True
        )
        if q_idx <9:
            st.markdown('<div data-testid="next_question">', unsafe_allow_html=True)
            if st.button("Next question", key="next_question"):
                st.session_state.current_q += 1
                st.session_state.answered = False
                st.session_state.selected_option = None
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            if st.session_state.total_questions > 0:
                percent_fake = (st.session_state.fake_tweet_count / 10) * 100

                st.markdown(f"### 🤖 AI-Generated Tweet Summary")
                st.markdown(f"🧠 **{percent_fake:.1f}%** of the tweets you saw were AI-generated.")

                if st.session_state.fake_tweets_list:
                    st.markdown("Here are the AI-generated tweets you saw:")
                    for fake_tweet in st.session_state.fake_tweets_list:
                        st.markdown(f"> {fake_tweet}")
            st.markdown("### 🏁 Final Results")
            st.balloons()
            score = st.session_state.score
            if score == 10:
                st.markdown(
    f"<div style='background-color: #fff3cd; color: black; padding: 12px; border-radius: 8px;'>  🎉 You scored {score}/10. WOW! </div>",
    unsafe_allow_html=True
)

            elif score >= 7:
                st.markdown(
    f"<div style='background-color: #d4edda; color: black; padding: 12px; border-radius: 8px;'>👏 Great job! You scored {score}/10!</div>",
    unsafe_allow_html=True
)
            elif score >= 4:
                st.markdown(
    f"<div style='background-color: #d1ecf1; color: black; padding: 12px; border-radius: 8px;'>ℹ️ You scored {score}/10. Not bad — give it another go!</div>",
    unsafe_allow_html=True
)

            else:
                st.markdown(
    f"<div style='background-color: #fff3cd; color: black; padding: 12px; border-radius: 8px;'>⚠️ You scored {score}/10. Tough round — try again! 💪</div>",
    unsafe_allow_html=True
)
            if st.button("🔁 Play again", key="play_again"):
                reset_game_state()
                st.session_state.page = "Home"
                st.rerun()




# GAME STARTED PAGE FUNCTION
def GameStarted():
    if st.session_state.game_mode == "easy":
        st.set_page_config(page_title="WHO SAID THAT? - Easy Mode", layout="wide")
    else:
        st.set_page_config(page_title="WHO SAID THAT? - Hard Mode", layout="wide")
    
    # Set light theme style (light yellow background, black text)
    set_light_theme()

    # Title and Content Container
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.title("🚀 GAME STARTED 🚀")

    st.markdown(f"""
    🎯 You have chosen the **{st.session_state.game_mode.capitalize()}** mode.
    🎮 You will be asked **10 questions**, and you have to guess the author of each tweet.
    """)

    st.markdown("Good luck! Let the game begin!")

    st.markdown('</div>', unsafe_allow_html=True)

    # Quit button in the top-right corner
    if st.button("Quit", key="quit"):
        st.session_state.page = "Home"
        st.rerun()
    
    #load the dataset
    df= load_data("data/tweets.csv")
    if df is None:
        st.error("Error loading data. Please check the file path.")
        return
    # Start the game based on the selected mode
    if st.session_state.game_mode == "easy":
        easy_question(df)
    else:
        st.write("THIS IS HARD MODE, WRITE FIRST AND LAST NAME OF THE AUTHOR")
        hard_question(df)
        
        



if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Navigation logic
if st.session_state.page == 'Home':
    Home()
else:
    GameStarted()
