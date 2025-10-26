# 🎬 StreamSense - Netflix Clone with AI Sentiment Analysis

A Netflix-style streaming platform clone with BERT-powered sentiment analysis for movie reviews.

## ✨ Features

- 🔐 **User Authentication** - Register/Login system with CSV storage
- 👨‍💼 **Admin Dashboard** - Analytics with Chart.js visualizations
- 🎥 **10 Netflix-style Movies** - Grid layout with hover effects
- 🤖 **AI Sentiment Analysis** - BERT model (92%+ accuracy) analyzes reviews
- 📊 **Real-time Analytics** - Positive/Negative sentiment distribution
- 🎨 **Netflix Dark Theme** - Beautiful UI with red highlights

## 🚀 Setup

### 1. Install Dependencies

```bash
pip install flask pandas torch torchvision transformers
```

Or use requirements.txt:

```bash
pip install -r requirements.txt
```

### 2. Ensure BERT Model Exists

Make sure `best_bert_sentiment_model.pth` is in the project root directory.

### 3. Run the Application

```bash
python app.py
```

The app will start on `http://localhost:5000`

## 👤 Login Credentials

### Admin Access
- **Email:** `admin@stream.com`
- **Password:** `admin123`
- **URL:** `http://localhost:5000/admin`

### Regular Users
Register a new account at `http://localhost:5000/register`

## 📁 Project Structure

```
BDML-MINI/
├── app.py                          # Flask backend
├── best_bert_sentiment_model.pth   # BERT model (required)
├── requirements.txt                # Dependencies
├── data/
│   ├── users.csv                   # User accounts
│   └── reviews.csv                 # Movie reviews
├── static/
│   ├── css/
│   │   └── style.css              # Netflix-style CSS
│   └── images/                     # Movie posters
├── templates/
│   ├── login.html                  # Login page
│   ├── register.html               # Registration page
│   ├── home.html                   # Movie grid homepage
│   ├── movie.html                  # Movie detail & reviews
│   ├── admin.html                  # Admin login
│   └── dashboard.html              # Admin dashboard
└── README.md
```

## 🎯 Usage

1. **Register** a new account
2. **Browse** 10 Netflix-style movies
3. **Click** on any movie to view details
4. **Write** a review - AI will analyze sentiment in real-time
5. **Admin** can view all reviews and sentiment statistics

## 🧠 AI Model

- **Model:** DistilBERT (66M parameters)
- **Accuracy:** 92.33%
- **Device:** CUDA GPU (if available)
- **Predictions:** Real-time sentiment analysis with confidence scores

## 📊 Admin Features

- View total reviews count
- Positive vs Negative distribution (Pie Chart)
- Most loved movie tracking
- Full reviews table with filters
- Export-ready analytics

## 🎨 Design

- Netflix dark theme (#141414)
- Red accent color (#E50914)
- Poppins font family
- Responsive grid layout
- Hover zoom effects
- Gradient overlays

## 🔧 Tech Stack

- **Backend:** Flask (Python)
- **Frontend:** HTML5, CSS3, JavaScript
- **AI:** PyTorch + Transformers (BERT)
- **Data:** Pandas + CSV
- **Charts:** Chart.js
- **Styling:** Custom Netflix-inspired CSS

## 📝 Notes

- Movie posters use placeholder images if files are missing
- CSV files auto-initialize on first run
- GPU acceleration for BERT (if available)
- Session-based authentication
- Real-time sentiment prediction

## 🐛 Troubleshooting

**Model not loading?**
- Ensure `best_bert_sentiment_model.pth` exists in root directory
- Check PyTorch and Transformers versions

**Port already in use?**
- Change port in `app.py`: `app.run(port=5001)`

**Import errors?**
- Activate virtual environment: `.venv\Scripts\activate`
- Reinstall dependencies: `pip install -r requirements.txt`

## 📜 License

Educational project - Free to use and modify

---

**Built with ❤️ using Flask + BERT**

