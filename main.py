from atproto import Client
from dotenv import load_dotenv
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from wordcloud import WordCloud

# Load environment variables from .env file
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

    # Preview first 5 followers
    for follower in followers_data[:5]:
        print(follower)

    # Save to CSV
    df = pd.DataFrame([vars(f) for f in followers_data])
    df.to_csv("followers_data.csv", index=False)
    print("Followers data saved to 'followers_data.csv'")
else:
    print("No followers found or error occurred.")


####Followers over time graph
# Load followers data from CSV
df = pd.read_csv("followers_data.csv")

# Handle invalid timestamps
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
df = df.dropna(subset=['created_at'])

# Ensure 'day' column contains datetime objects
df['day'] = df['created_at'].dt.floor('D')
followers_per_day = df.groupby('day').size()
cumulative_followers = followers_per_day.cumsum()

# Set the index to datetime for proper plotting
cumulative_followers.index = pd.to_datetime(cumulative_followers.index)

# Plot cumulative followers over time
plt.figure(figsize=(12, 8))
plt.plot(cumulative_followers.index, cumulative_followers, marker='o', label='Cumulative Followers')

# Customize x-axis for monthly labels
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

# Improve tick labels
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)

# Add titles and grid
plt.title("Cumulative Followers Over Time", fontsize=16, fontweight='bold')
plt.grid(visible=True, linestyle='--', alpha=0.6)

# Add legend
plt.legend(fontsize=10, loc='upper left')

plt.tight_layout()
plt.show()


###wordcloud folowers info
# Combine all descriptions into a single string
descriptions = ' '.join(df['description'].dropna().astype(str))

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(descriptions)

# Plot the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title(" ", fontsize=16, fontweight='bold')
plt.show()


# Convert 'created_at' to datetime for followers
df['follower_created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

# Plot a histogram of account creation dates per month
plt.figure(figsize=(12, 8))
df['follower_created_at'].dt.to_period('M').value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title("Distribution of Follower Account Creation Dates (Monthly)", fontsize=16, fontweight='bold')
plt.xlabel("Month of Account Creation")
plt.ylabel("Number of Followers")
plt.xticks(rotation=45)
plt.grid(visible=True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# Count followers with and without custom avatars
has_avatar = df['avatar'].notna().sum()
no_avatar = len(df) - has_avatar

# Plot a pie chart with enhanced visualization
plt.figure(figsize=(8, 6))
plt.pie(
    [has_avatar, no_avatar], 
    labels=["Has Avatar", "No Avatar"], 
    autopct='%1.1f%%', 
    colors=['#1f77b4', '#ff7f0e'], 
    startangle=140, 
    wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}
)
plt.title("Custom Avatars vs. Default Avatars", fontsize=16, fontweight='bold')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.tight_layout()
plt.show()

# Calculate daily growth rate
daily_growth = followers_per_day.pct_change().fillna(0) * 100  # Percent change

# Plot growth rate
plt.figure(figsize=(12, 8))
daily_growth.plot(kind='bar', color='salmon', alpha=0.8)
plt.title("Daily Follower Growth Rate (%)", fontsize=16, fontweight='bold')
plt.xlabel("Day")
plt.ylabel("Growth Rate (%)")
plt.xticks(rotation=45)
plt.grid(visible=True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

