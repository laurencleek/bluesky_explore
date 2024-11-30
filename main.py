from atproto import Client
from dotenv import load_dotenv
import os
import matplotlib.dates as mdates
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from genderize import Genderize
import pandas as pd
import matplotlib.pyplot as plt
import gender_guesser.detector as gender
import numpy as np

# Load environment variables from .env file (gitignored but just create a .env file with your handle and password)
load_dotenv(dotenv_path=r"C:\Users\lclee\OneDrive - Istituto Universitario Europeo\bluesky\.env")

# Get credentials from environment variables
handle = os.getenv("HANDLE")
app_password = os.getenv("APP_PASSWORD")

# Authenticate with the API
client = Client()

try:
    client.login(handle, app_password)
    print("Login successful!")
except Exception as e:
    print(f"Failed to log in: {e}")
    exit()

# Fetch Followers
def fetch_followers(client, actor_handle):
    followers = []
    cursor = None

    try:
        while True:
            print(f"Fetching followers with cursor: {cursor}")
            response = client.get_followers(actor=actor_handle, cursor=cursor)
            followers.extend(response.followers)
            cursor = response.cursor
            if not cursor:
                break
        return followers
    except Exception as e:
        print(f"Error fetching followers: {e}")
        return []

followers_data = fetch_followers(client, "laurenleek.eu")

if followers_data:
    print(f"Fetched {len(followers_data)} followers")

    for follower in followers_data[:5]:
        print(follower)

    # Save to CSV
    df = pd.DataFrame([vars(f) for f in followers_data])
    df.to_csv("followers_data.csv", index=False)
    print("Followers data saved to 'followers_data.csv'")
else:
    print("No followers found or error occurred.")


### Followers over time graph
df = pd.read_csv("followers_data.csv")
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
df = df.dropna(subset=['created_at'])
df['day'] = df['created_at'].dt.floor('D')
followers_per_day = df.groupby('day').size()
cumulative_followers = followers_per_day.cumsum()
cumulative_followers.index = pd.to_datetime(cumulative_followers.index)
plt.figure(figsize=(12, 8))
plt.plot(cumulative_followers.index, cumulative_followers, marker='o', label='Cumulative Followers')
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.title("Cumulative Followers Over Time", fontsize=16, fontweight='bold')
plt.grid(visible=True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10, loc='upper left')
plt.tight_layout()
plt.show()

### Wordcloud followers info
descriptions = ' '.join(df['description'].dropna().astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(descriptions)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title(" ", fontsize=16, fontweight='bold')
plt.show()

### Plot a histogram of account creation dates per month
df['follower_created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
plt.figure(figsize=(12, 8))
df['follower_created_at'].dt.to_period('M').value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title("Distribution of Follower Account Creation Dates (Monthly)", fontsize=16, fontweight='bold')
plt.xlabel("Month of Account Creation")
plt.ylabel("Number of Followers")
plt.xticks(rotation=45)
plt.grid(visible=True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


### Create themes over time graph
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
df = df.dropna(subset=['created_at'])
df['day'] = df['created_at'].dt.floor('D')
vectorizer = CountVectorizer(stop_words='english', max_features=20)
word_counts = vectorizer.fit_transform(df['description'].dropna())
words_df = pd.DataFrame(word_counts.toarray(), columns=vectorizer.get_feature_names_out())
words_df['day'] = df['day'].values[:len(words_df)]  # Ensure matching dimensions for 'day'
excluded_words = ['delete', 'bsky', 'com', 'https', 'www']
filtered_words_df = words_df.drop(columns=[word for word in excluded_words if word in words_df])
filtered_words_over_time = filtered_words_df.groupby('day').sum()
themes = {
    "Economics": ["economics", "economy", "policy"],
    "University": ["phd", "professor", "university", "research", "researcher"],
    "Politics": ["politics", "political", "social"],
    "Climate": ["climate"],
    "Europe": ["eu", "european"]
}
grouped_themes = pd.DataFrame(index=filtered_words_over_time.index)
for theme, words in themes.items():
    grouped_themes[theme] = filtered_words_over_time[words].sum(axis=1)
plt.figure(figsize=(14, 8))
grouped_themes.plot.area(ax=plt.gca(), alpha=0.8, colormap='tab20c')  # Use a colormap for better distinction
plt.title("Word Trends in Follower Descriptions Grouped by Theme (Stacked)", fontsize=16, fontweight='bold')
plt.ylabel("Frequency")
plt.legend(title="Themes", fontsize=10, loc='upper left')
plt.grid(visible=True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

### Gender comparison graph
df['first_name'] = df['display_name'].str.split().str[0]  # Extract first word as the first name

d = gender.Detector()
df['predicted_gender'] = df['first_name'].apply(lambda x: d.get_gender(x) if isinstance(x, str) else None)
df['filtered_gender'] = df['predicted_gender'].replace(
    {"mostly_male": "male", "mostly_female": "female", "andy": None, "unknown": None}
)

df_cleaned = df.dropna(subset=['filtered_gender'])

gender_totals = df_cleaned['filtered_gender'].value_counts()
plt.figure(figsize=(8, 6))
gender_totals.plot.bar(color=["#1f77b4", "#ff7f0e"], alpha=0.85, edgecolor="black", linewidth=1.2)
for index, value in enumerate(gender_totals):
    plt.text(
        index,
        value + 0.5,
        f"{int(value):,}",
        ha="center",
        fontsize=12,
        fontweight="bold"
    )

plt.title("Total Gender Distribution of Followers", fontsize=18, fontweight="bold")
plt.ylabel("Number of Followers", fontsize=14)
plt.xticks(rotation=0, fontsize=12, fontweight="bold")
plt.yticks(fontsize=12)
plt.grid(visible=True, linestyle="--", alpha=0.6, axis="y")
plt.tight_layout()
plt.show()

### Academic affiliation by gender before and after October graph
academic_ranks = {
    "Student": ["student", "undergraduate", "bachelor", "master", "msc"],
    "PhD": ["phd", "doctoral", "doctorate"],
    "Professor": ["professor", "lecturer", "faculty"]
}
df['academic_rank'] = None
for rank, keywords in academic_ranks.items():
    df.loc[df['description'].str.contains('|'.join(keywords), case=False, na=False), 'academic_rank'] = rank

df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce').dt.tz_localize(None)
cutoff_date = pd.Timestamp('2024-10-31')
df['time_period'] = df['created_at'].apply(
    lambda x: 'Before October' if x <= cutoff_date else 'After October'
)
rank_gender_time_distribution = df.groupby(['time_period', 'academic_rank', 'filtered_gender']).size().unstack(fill_value=0)
rank_gender_time_distribution = rank_gender_time_distribution.loc[
    pd.IndexSlice[:, ["Student", "PhD", "Professor"]], :
]

labels = ["Student", "PhD", "Professor"]
time_periods = rank_gender_time_distribution.index.get_level_values('time_period').unique()
x = np.arange(len(labels))
width = 0.2

colors = {
    "Before October": {"male": "#4e79a7", "female": "#f28e2c"},
    "After October": {"male": "#76b7b2", "female": "#e15759"}
}

fig, ax = plt.subplots(figsize=(14, 8))
for i, time_period in enumerate(time_periods):
    for j, gender in enumerate(["male", "female"]):
        ax.bar(
            x + (i * 2 + j) * width - width * 1.5,
            rank_gender_time_distribution.loc[time_period, gender],
            width,
            label=f"{time_period} ({gender.capitalize()})" if (i == 0 and j == 0) else None,
            color=colors[time_period][gender],
            edgecolor="black"
        )

for i, time_period in enumerate(time_periods):
    for j, gender in enumerate(["male", "female"]):
        for idx, val in enumerate(rank_gender_time_distribution.loc[time_period, gender]):
            ax.text(
                x[idx] + (i * 2 + j) * width - width * 1.5,
                val + 0.5,
                f"{int(val)}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold"
            )

ax.set_title("Academic Ranks by Gender Before and After October 2024", fontsize=16, fontweight="bold")
ax.set_ylabel("Number of Followers", fontsize=12)
ax.set_xlabel("Academic Rank", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12, fontweight="bold")
legend_elements = [
    plt.Line2D([0], [0], color=colors["Before October"]["male"], lw=6, label="Before October (Male)"),
    plt.Line2D([0], [0], color=colors["Before October"]["female"], lw=6, label="Before October (Female)"),
    plt.Line2D([0], [0], color=colors["After October"]["male"], lw=6, label="After October (Male)"),
    plt.Line2D([0], [0], color=colors["After October"]["female"], lw=6, label="After October (Female)")
]
ax.legend(handles=legend_elements, fontsize=10, loc="upper left", title="Time Period (Gender)")
ax.grid(visible=True, linestyle="--", alpha=0.6, axis="y")
plt.tight_layout()
plt.show()

