# AIHive Coach MVP v0.1

**Think Clearly. Act Confidently. Grow Daily.**

An AI personal development and goal-setting coach that guides users through structured coaching conversations, helps them create concrete action plans, and tracks progress over time.

## 🎯 Product Overview

AIHive Coach is built on a GROW-inspired coaching framework that:
- **Clarifies goals** through thoughtful questions
- **Explores reality** by distinguishing facts from assumptions
- **Generates options** by challenging limiting beliefs
- **Builds commitment** through concrete action plans

The coach is supportive but not flattering, insightful but not prescriptive, and always respects user autonomy.

## ✨ Features

### 1. Coaching Conversations
- Clean, responsive chat interface
- Four coaching categories: Career & Leadership, Personal Growth, Goals & Productivity, Decision Making
- Adaptive questioning based on GROW framework
- Crisis detection and appropriate resource referrals

### 2. Action Plans
- Generate structured action plans from conversations
- Five key components: Goal, Key Insight, Next Action, Target Date, Success Indicator
- Editable and saveable

### 3. Progress Reviews
- Return to previous action plans
- Add progress notes
- Track completion status

## 🏗️ Architecture

### Tech Stack
- **Frontend**: Streamlit (Python)
- **AI**: OpenAI API (GPT-4o-mini) with Demo Mode fallback
- **Database**: SQLite (development)
- **Data Models**: Pydantic
- **ORM**: SQLAlchemy

### Directory Structure
```
aihive_coach/
├── app.py                  # Main Streamlit app
├── config.py              # Configuration management
├── database.py            # SQLite database layer
├── coach_ai.py            # AI coaching logic
├── prompts.py             # System prompts
├── models.py              # Pydantic data models
├── requirements.txt       # Python dependencies
├── pages/
│   ├── coaching.py        # Coaching workspace
│   └── action_plans.py    # Action plan management
├── static/
│   ├── styles.css         # Custom CSS
│   └── i18n/
│       ├── en.json        # English strings
│       └── th.json        # Thai placeholders
└── tests/
    ├── test_coach_ai.py   # Unit tests
    ├── test_database.py   # Database tests
    └── test_e2e.py        # End-to-end tests
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- OpenAI API key (optional - will use Demo Mode without it)

### Installation

1. **Clone and navigate to the directory**:
   ```bash
   cd aihive_coach
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment** (optional):
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

4. **Run the app**:
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** to `http://localhost:8501`

### Demo Mode

If no `OPENAI_API_KEY` is provided, the app automatically runs in **Demo Mode** using heuristic responses. This allows full testing of the UI and flow without API costs.

## 🧪 Testing

### Run Unit Tests
```bash
pytest tests/test_coach_ai.py tests/test_database.py -v
```

### Run End-to-End Tests
```bash
# Install Playwright browsers (first time only)
playwright install chromium

# Run E2E tests
pytest tests/test_e2e.py -v
```

### Test Coverage
- Starting a coaching session
- Sending/receiving messages
- Creating and editing action plans
- Progress reviews
- Mobile responsiveness
- Cross-user isolation
- Crisis detection

## 🔒 Privacy & Security

- **User Isolation**: Each user has a unique anonymous ID; users cannot access each other's data
- **API Keys**: Stored server-side via environment variables, never exposed to client
- **Crisis Handling**: Detects self-harm language and provides appropriate resources
- **Boundaries**: Clearly disclaims not being a licensed therapist

## 💰 Cost Estimation

### Model: GPT-4o-mini
- Input: $0.00015 per 1K tokens
- Output: $0.0006 per 1K tokens

### Per-Session Estimate
- Average conversation: ~3,000 input + 1,500 output tokens
- **Cost per session**: ~$0.0014 (less than 1 cent)

### Monthly Projections
- **100 users** (5 sessions/month each): ~$0.70/month
- **1,000 users** (5 sessions/month each): ~$7.00/month
- **10,000 users** (5 sessions/month each): ~$70/month

*Note: Costs can be optimized by using context pruning, caching, or switching models.*

## 🌐 Internationalization

Currently English-only, but architected for i18n:
- All UI strings in `static/i18n/en.json`
- Thai placeholder file included (`th.json`)
- Ready for multi-language support

## 📋 Limitations & Future Work

### Current MVP Limitations
- No authentication (anonymous browser-based IDs)
- SQLite database (not production-ready for scale)
- No payment/subscription system
- No coach effectiveness analytics
- No conversation export
- No push notifications for action plan deadlines

### Recommended Next Development Cycle
1. **User accounts** with email/password authentication
2. **PostgreSQL** or cloud database for production
3. **Coach effectiveness metrics**: track user satisfaction, action plan completion rates
4. **A/B test** coaching prompts and question strategies
5. **API cost monitoring** dashboard
6. **Email reminders** for action plan target dates
7. **Conversation history** search and export
8. **Willingness-to-pay** research with pricing tiers

## 📄 License

© 2026 AIHiveHQ. Founded by Rungnapa Mukdasanit.

## 🆘 Support

For questions or issues:
- Founder: Rungnapa Mukdasanit
- Website: https://aihivehq.com (coming soon)
