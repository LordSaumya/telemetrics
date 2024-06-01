import pandas as pd
from textblob import TextBlob
import plotly.express as px
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import silhouette_score
import random

def analyse_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Analyse the data and return a dictionary of results

    Parameters:
    df: pd.DataFrame - The data to analyse

    Returns:
    dict[str, pd.DataFrame] - A dictionary of results
    """
    # Add new columns to df
    df["date"] = pd.to_datetime(df["timestamp"], unit="s") + pd.Timedelta("08:00:00")
    df["day"] = df["date"].dt.day_name()
    df["month"] = df["date"].dt.month
    df["hour"] = df["date"].dt.hour
    df["difference"] = df["date"].diff().dt.total_seconds().abs()
    df["sentiment"] = df["content"].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Messages DF
    messages = df

    # Statistics DF
    stats = pd.DataFrame()
    stats["Total Messages"] = [df.shape[0]]
    stats["Average Messages per Day"] = stats["Total Messages"] / (df["date"].max() - df["date"].min()).days
    stats["Median difference between messages"] = df["difference"].median()
    stats["Median difference between replies"] = df[df["name"] != df["name"].shift(1)]["difference"].median()
    stats["Average message length"] = df["content"].str.len().mean()
    stats["Average sentiment"] = df["sentiment"].mean()

    # Messages by person
    for name in df["name"].unique():
        stats[f"{name}"] = df["name"].value_counts()[name]
    
    # Most common words by person
    split_into_words = lambda text: text.lower().split(" ")
    exploded_df = df.assign(words=df['content'].apply(split_into_words)).explode('words')
    
    # Group by words and name, then count occurrences
    most_common_words_by_person = exploded_df.groupby(['words', 'name']).size().reset_index(name='count')
    most_common_words_by_person.columns = ['word', 'name', 'count']

    # Assign a new column with total count to each word
    most_common_words_by_person['total_count'] = most_common_words_by_person.groupby('word')['count'].transform('sum')

    # Sort by total count
    most_common_words_by_person = most_common_words_by_person[most_common_words_by_person["word"].str.len() > 4].sort_values(by='total_count', ascending=False).head(20)

    # Positive Sentiment DF (5 most positive messages by each person)
    pos_sent_df = df.groupby("name").apply(lambda x: x.nlargest(5, "sentiment")).reset_index(drop=True)

    # Negative Sentiment DF (5 most negative messages by each person)
    neg_sent_df = df.groupby("name").apply(lambda x: x.nsmallest(5, "sentiment")).reset_index(drop=True)

    return {"messages": messages, "stats": stats, "most_common_words": most_common_words_by_person, "pos_sent_df": pos_sent_df, "neg_sent_df": neg_sent_df}


def create_graphs(dfs: dict[str, pd.DataFrame]) -> list:
    """
    Creates and returns plotly graphs and tables from analysis results

    Parameters:
    dfs: dict[str, pd.DataFrame] - A dictionary of DataFrames containing analysis results
    
    Returns:
    list - A list of plotly graphs
    """
    
    # Plot 1: Pie chart of total messages by person
    fig1 = px.pie(dfs["stats"].drop(["Total Messages", "Average Messages per Day", "Median difference between messages", "Median difference between replies", "Average message length", "Average sentiment"], axis=1).transpose(), names=dfs["stats"].drop(["Total Messages", "Average Messages per Day", "Median difference between messages", "Median difference between replies", "Average message length", "Average sentiment"], axis=1).columns, values=dfs["stats"].drop(["Total Messages", "Average Messages per Day", "Median difference between messages", "Median difference between replies", "Average message length", "Average sentiment"], axis=1).values[0], title="Total Messages by Person", template="plotly_dark")

    # Plot 2: Number of messages by day of the week
    fig2 = px.bar(dfs["messages"]["day"].value_counts().reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).reset_index(), x="day", y="count", title="Number of Messages by Day of the Week", labels={"day": "Day of the Week", "count": "Number of Messages"}, template="seaborn")

    # Plot 3: Number of messages by month
    fig3 = px.bar(dfs["messages"]["month"].value_counts().sort_index().reset_index(), x="month", y="count", title="Number of Messages by Month", labels={"month": "Month", "count": "Number of Messages"}, template="seaborn")

    # Plot 4: Number of messages by hour of the day
    fig4 = px.bar(dfs["messages"]["hour"].value_counts().sort_index().reset_index(), x="hour", y="count", title="Number of Messages by Hour of the Day", labels={"hour": "Hour of the Day", "count": "Number of Messages"}, template="plotly_dark")

    # Plot 5: Time between messages
    fig5 = px.histogram(dfs["messages"], x="difference", title="Time between Messages", labels={"difference": "Time between Messages (sec)", "count": "Number of Messages"}, template="plotly_dark")

    # Plot 6: Sentiment Analysis
    fig6 = px.histogram(dfs["messages"], x="sentiment", title="Sentiment Analysis", labels={"sentiment": "Sentiment", "count": "Number of Messages"}, template="seaborn")

    # Plot 7: Most common words stacked bar chart by person
    fig7 = px.bar(dfs["most_common_words"], x="word", y="count", color="name", title="Most Commonly-Used Words", labels={"word": "Word", "count": "Count", "name": "Person"}, template = "seaborn")

    # Plot 8: Heatmap of messages by day and hour
    df_heatmap = dfs["messages"].groupby(["day", "hour"]).size().reset_index(name='count')
    fig8 = px.density_heatmap(df_heatmap, x="hour", y="day", z="count", title="Heatmap of Messages by Hour and Day", 
                          labels={"hour": "Hour of the Day", "day": "Day of the Week", "count": "Messages"}, 
                          category_orders={"day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]}, nbinsx=12, nbinsy=7,
                          template="plotly_dark")
    fig8.update_layout(xaxis = dict(dtick=2))

    # Plot 9: Bubble plot of clusters (x and y positions are random, size is the count of messages in the cluster)
    # Remove x and y axes and gridlines, plot labels on top of bubbles
    # Increased scale of bubbles for better visibility
    top_words = clusters(dfs["messages"])
    if top_words.shape[0] == 4:
        x_pos = [0, 100, 0, 100]
        y_pos = [0, 0, 100, 100]
    else:
        x_pos = [0, 100, 0, 100, 0]
        y_pos = [0, 0, 100, 100, 100]
    x_pos = [x + random.uniform(-25, 25) for x in x_pos]
    y_pos = [y + random.uniform(-25, 25) for y in y_pos]
    fig9 = px.scatter(top_words, x=x_pos, y=y_pos, size="count", hover_name="top_words", title="Word Clusters", template="plotly_dark", size_max=90, text = "top_words", color = [x_pos[i] + y_pos[i] for i in range(len(x_pos))])
    fig9.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))
    fig9.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, visible = False)
    fig9.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, visible = False)
    fig9.update_layout(showlegend=False, coloraxis_showscale=False)

    # Plot 10: Bar chart of message length by person
    fig10 = px.bar(dfs["messages"].groupby("name")["content"].apply(lambda x: x.str.len().mean()).reset_index(name="average_length"), x="name", y="average_length", title="Average Message Length by Person", labels={"name": "Person", "average_length": "Average Message Length (characters)"}, template="seaborn")

    return [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10]

def rel_score(df: pd.DataFrame, stats: pd.DataFrame) -> float:
    """
    Calculate the relationship score between two people based on the number of messages sent and sentiment analysis.

    Parameters:
    df: pd.DataFrame - The messages
    stats: pd.DataFrame - The analysed statistics of the messages

    Returns:
    float - The relationship score, between -1 (enemies) and 1 (lovers)
    float - The message score, between 0 and 1
    float - The time span score, between 0 and 1
    float - The reply score, between 0 and 1
    """

    clamp = lambda n : -1 if n < -1 else 1 if n > 1 else n

    # Calculate the relationship score
    sentiment_score = clamp(df[df["sentiment"] != 0]["sentiment"].mean() * 4)

    time_span = (df["date"].max() - df["date"].min()).days
    time_span_score = clamp(time_span / 365)

    message_score = clamp(df.shape[0] / (15 * time_span))

    reply_score = clamp((1/(stats["Median difference between replies"][0] / 7200))/100)
    
    if sentiment_score < 0:
        relationship_score = 0.2 * (1/message_score) + 0.1 * time_span_score + 0.2 * (1/reply_score) + 0.5 * sentiment_score
    else:
        relationship_score = 0.2 * message_score + 0.1 * time_span_score + 0.2 * reply_score + 0.5 * sentiment_score

    return round(relationship_score, 2), round(message_score, 2), round(time_span_score, 2), round(reply_score, 2), round(sentiment_score, 2)

def clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform clustering on the messages data and return a DataFrame with the clusters

    Parameters:
    df: pd.DataFrame - The data to cluster

    Returns:
    pd.DataFrame - counts and top words for each cluster
    """
    df = df.copy()
    nltk.download('stopwords')
    nltk.download('wordnet')
    lemmatiser = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Preprocess data
    def preprocess_msg(msg):
        msg = re.sub(r'[^a-zA-Z\s]', '', msg).lower()
        tokens = msg.split()
        words_to_filter = ['think', 'kinda', 'thanks', 'thing', 'wanna', 'thats', 'probably', 'coming', 'today', 'didnt', 'maybe', 'later', 'gonna', 'really', 'hahaha', 'about', 'aight', 'sorry', 'going', 'hahahaha', 'right', 'actually', 'whats', 'havent', 'youre', 'gotta']
        tokens = filter(lambda x: len(x) > 4 and x not in words_to_filter, [lemmatiser.lemmatize(word) for word in tokens if word not in stop_words])
        return ' '.join(tokens)
    df['content'] = df['content'].apply(preprocess_msg)
    df = df[df['content'].str.len() > 4]

    # Vectorise data
    vectoriser = CountVectorizer(max_features=1000, stop_words='english')
    X = vectoriser.fit_transform(df['content'])

    # Compute optimal number of clusters using silhouette score
    scores = []
    for i in range(4, 6):
        lda = LatentDirichletAllocation(n_components=i, random_state=0)
        lda.fit(X)
        score = silhouette_score(X, lda.transform(X).argmax(axis=1))
        scores.append(score)
    n_clusters = scores.index(max(scores)) + 4

    # Perform clustering
    lda = LatentDirichletAllocation(n_components=n_clusters)
    lda.fit(X)
    df['cluster'] = lda.transform(X).argmax(axis=1)

    def get_top_words(model, feature_names, n_top_words):
        top_words = {}
        for topic_idx, topic in enumerate(model.components_):
            top_words[topic_idx] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        return top_words

    # Get feature names
    feature_names = vectoriser.get_feature_names_out()
    top_words = get_top_words(lda, feature_names, 5)  # Adjust the number of top words

    counts = df['cluster'].value_counts().reset_index()
    counts.columns = ['cluster', 'count']
    counts['top_words'] = counts['cluster'].map(top_words)

    return counts[['top_words', 'count']].sort_values(by='count', ascending=False)
