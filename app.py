from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import pandas as pd
import os
from datetime import datetime
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import re
import json

app = Flask(__name__)
app.secret_key = 'streamsense_secret_key_2024'

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('static/images', exist_ok=True)
os.makedirs('model', exist_ok=True)

# Load BERT model for sentiment analysis
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading BERT model on {device}...")

try:
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    model.load_state_dict(torch.load('model/best_bert_sentiment_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
    print("✅ BERT model loaded successfully!")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    model = None
    tokenizer = None

# Initialize CSV files if they don't exist
if not os.path.exists('data/users.csv'):
    pd.DataFrame(columns=['username', 'email', 'password']).to_csv('data/users.csv', index=False)

if not os.path.exists('data/reviews.csv'):
    pd.DataFrame(columns=['username', 'movie_title', 'review_text', 'sentiment', 'timestamp']).to_csv('data/reviews.csv', index=False)

# Movie database (with Netflix-style movies)
MOVIES = [
    {'id': 1, 'title': 'Stranger Things', 'poster': 'stranger_things.jpg', 'description': 'A group of friends uncover supernatural mysteries in their small town.'},
    {'id': 2, 'title': 'The Crown', 'poster': 'the_crown.jpg', 'description': 'The life of Queen Elizabeth II and the royal family.'},
    {'id': 3, 'title': 'Money Heist', 'poster': 'money_heist.jpg', 'description': 'Eight thieves take hostages in the Royal Mint of Spain.'},
    {'id': 4, 'title': 'Breaking Bad', 'poster': 'breaking_bad.jpg', 'description': 'A chemistry teacher turned methamphetamine manufacturer.'},
    {'id': 5, 'title': 'The Witcher', 'poster': 'the_witcher.jpg', 'description': 'A monster hunter struggles to find his place in a world.'},
    {'id': 6, 'title': 'Dark', 'poster': 'dark.jpg', 'description': 'A family saga with a supernatural twist set in a German town.'},
    {'id': 7, 'title': 'Narcos', 'poster': 'narcos.jpg', 'description': 'The rise and fall of Colombian drug lord Pablo Escobar.'},
    {'id': 8, 'title': 'Squid Game', 'poster': 'squid_game.jpg', 'description': 'Hundreds compete in deadly children\'s games for a prize.'},
    {'id': 9, 'title': 'Wednesday', 'poster': 'wednesday.jpg', 'description': 'Wednesday Addams investigates mysteries at Nevermore Academy.'},
    {'id': 10, 'title': 'Black Mirror', 'poster': 'black_mirror.jpg', 'description': 'Anthology series exploring dark and satirical themes about technology.'}
]

# Admin credentials
ADMIN_EMAIL = 'admin@stream.com'
ADMIN_PASSWORD = 'admin123'

def clean_text(text):
    """Clean text for sentiment analysis"""
    text = re.sub(r'<br\s*/?\s*>', ' ', text)
    text = re.sub(r'http\S+', '', text)
    return text.strip()

def predict_sentiment(text):
    """Predict sentiment using BERT model"""
    if model is None or tokenizer is None:
        return "Neutral", 0.5
    
    text = clean_text(text)
    
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
        probs = torch.softmax(logits, dim=1)[0]
    
    sentiment = "Positive" if pred == 1 else "Negative"
    confidence = probs[pred].item()
    
    return sentiment, confidence

# Routes
@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('home'))
    if 'admin' in session:
        return redirect(url_for('admin_dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if admin
        if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
            session['admin'] = True
            return redirect(url_for('admin_dashboard'))
        
        # Check regular user
        try:
            users_df = pd.read_csv('data/users.csv')
            user = users_df[(users_df['email'] == email) & (users_df['password'] == password)]
            
            if not user.empty:
                session['user'] = user['username'].values[0]
                return redirect(url_for('home'))
            else:
                flash('Invalid email or password', 'error')
        except Exception as e:
            flash(f'Error: {str(e)}', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        try:
            users_df = pd.read_csv('data/users.csv')
            
            # Check if user already exists
            if email in users_df['email'].values:
                flash('Email already registered', 'error')
                return render_template('register.html')
            
            # Add new user
            new_user = pd.DataFrame([[username, email, password]], columns=['username', 'email', 'password'])
            users_df = pd.concat([users_df, new_user], ignore_index=True)
            users_df.to_csv('data/users.csv', index=False)
            
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            flash(f'Error: {str(e)}', 'error')
    
    return render_template('register.html')

@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('home.html', movies=MOVIES, username=session['user'])

@app.route('/movie/<int:movie_id>', methods=['GET', 'POST'])
def movie(movie_id):
    if 'user' not in session:
        return redirect(url_for('login'))
    
    movie = next((m for m in MOVIES if m['id'] == movie_id), None)
    if not movie:
        flash('Movie not found', 'error')
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        review_text = request.form.get('review')
        
        if review_text and len(review_text.strip()) > 0:
            # Predict sentiment (but don't show to user)
            sentiment, confidence = predict_sentiment(review_text)
            
            # Save review
            try:
                reviews_df = pd.read_csv('data/reviews.csv')
                new_review = pd.DataFrame([[
                    session['user'],
                    movie['title'],
                    review_text,
                    sentiment,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ]], columns=['username', 'movie_title', 'review_text', 'sentiment', 'timestamp'])
                
                reviews_df = pd.concat([reviews_df, new_review], ignore_index=True)
                reviews_df.to_csv('data/reviews.csv', index=False)
                
                flash('Thank you! Your review has been submitted successfully.', 'success')
            except Exception as e:
                flash(f'Error saving review: {str(e)}', 'error')
        else:
            flash('Please enter a review', 'error')
    
    # Load existing reviews for this movie
    try:
        reviews_df = pd.read_csv('data/reviews.csv')
        movie_reviews = reviews_df[reviews_df['movie_title'] == movie['title']].to_dict('records')
    except:
        movie_reviews = []
    
    return render_template('movie.html', movie=movie, reviews=movie_reviews)

@app.route('/admin', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
            session['admin'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid admin credentials', 'error')
    
    return render_template('admin.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    if 'admin' not in session:
        return redirect(url_for('admin_login'))
    
    selected_movie = request.args.get('movie', 'all')
    
    try:
        reviews_df = pd.read_csv('data/reviews.csv')
        
        # Get list of movies for dropdown
        movie_list = ['all'] + sorted(reviews_df['movie_title'].unique().tolist())
        
        # Filter by selected movie
        if selected_movie != 'all':
            filtered_df = reviews_df[reviews_df['movie_title'] == selected_movie]
        else:
            filtered_df = reviews_df
        
        # Calculate statistics
        total_reviews = len(filtered_df)
        positive_count = len(filtered_df[filtered_df['sentiment'] == 'Positive'])
        negative_count = len(filtered_df[filtered_df['sentiment'] == 'Negative'])
        
        # Calculate accuracy (assuming we have ground truth - for demo purposes)
        accuracy = 92.33  # From your BERT model training
        
        # Most positive movie(s) - show all movies with max positive reviews
        if not reviews_df.empty and selected_movie == 'all':
            positive_movies = reviews_df[reviews_df['sentiment'] == 'Positive']['movie_title'].value_counts()
            if len(positive_movies) > 0:
                max_count = positive_movies.max()
                most_positive_list = positive_movies[positive_movies == max_count].index.tolist()
                most_positive = ', '.join(most_positive_list)
            else:
                most_positive = 'N/A'
        else:
            most_positive = 'N/A' if selected_movie == 'all' else selected_movie
        
        reviews = filtered_df.to_dict('records')
        
        stats = {
            'total': total_reviews,
            'positive': positive_count,
            'negative': negative_count,
            'most_positive': most_positive,
            'accuracy': accuracy,
            'selected_movie': selected_movie
        }
        
    except Exception as e:
        reviews = []
        movie_list = ['all']
        stats = {'total': 0, 'positive': 0, 'negative': 0, 'most_positive': 'N/A', 'accuracy': 0, 'selected_movie': 'all'}
    
    return render_template('dashboard.html', reviews=reviews, stats=stats, movies=movie_list)

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('login'))

@app.route('/api/sentiment-stats')
def sentiment_stats():
    """API endpoint for chart data - positive vs negative for specific movie"""
    selected_movie = request.args.get('movie', 'all')
    
    try:
        reviews_df = pd.read_csv('data/reviews.csv')
        
        if selected_movie != 'all':
            reviews_df = reviews_df[reviews_df['movie_title'] == selected_movie]
        
        positive = len(reviews_df[reviews_df['sentiment'] == 'Positive'])
        negative = len(reviews_df[reviews_df['sentiment'] == 'Negative'])
        return jsonify({'positive': positive, 'negative': negative})
    except:
        return jsonify({'positive': 0, 'negative': 0})

@app.route('/api/movie-distribution')
def movie_distribution():
    """API endpoint for movie distribution chart"""
    try:
        reviews_df = pd.read_csv('data/reviews.csv')
        
        # Get positive and negative counts per movie
        movie_stats = {}
        for movie in reviews_df['movie_title'].unique():
            movie_reviews = reviews_df[reviews_df['movie_title'] == movie]
            positive = len(movie_reviews[movie_reviews['sentiment'] == 'Positive'])
            negative = len(movie_reviews[movie_reviews['sentiment'] == 'Negative'])
            movie_stats[movie] = {'positive': positive, 'negative': negative, 'total': positive + negative}
        
        return jsonify(movie_stats)
    except:
        return jsonify({})

@app.route('/api/wordcloud-data')
def wordcloud_data():
    """API endpoint for word cloud data"""
    selected_movie = request.args.get('movie', 'all')
    sentiment_type = request.args.get('type', 'positive')
    
    try:
        reviews_df = pd.read_csv('data/reviews.csv')
        
        if selected_movie != 'all':
            reviews_df = reviews_df[reviews_df['movie_title'] == selected_movie]
        
        # Filter by sentiment
        sentiment = 'Positive' if sentiment_type == 'positive' else 'Negative'
        filtered = reviews_df[reviews_df['sentiment'] == sentiment]
        
        # Combine all review texts
        all_text = ' '.join(filtered['review_text'].tolist())
        
        # Simple word frequency (you can enhance this)
        words = all_text.lower().split()
        from collections import Counter
        word_freq = Counter(words)
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'it', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
        
        word_freq = {word: count for word, count in word_freq.items() if word not in stop_words and len(word) > 3}
        
        # Get top 50 words
        top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:50])
        
        return jsonify(top_words)
    except Exception as e:
        return jsonify({})

if __name__ == '__main__':
    print("\n🎬 StreamSense is starting...")
    print("=" * 50)
    print("👤 Admin Login:")
    print(f"   Email: {ADMIN_EMAIL}")
    print(f"   Password: {ADMIN_PASSWORD}")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)

