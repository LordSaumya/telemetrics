import pandas as pd
from textblob import TextBlob
import plotly.express as px
import re


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
     
    # Plot 1: Number of messages by day of the week
    fig1 = px.bar(dfs["messages"]["day"].value_counts().reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).reset_index(), x="day", y="count", title="Number of Messages by Day of the Week", labels={"day": "Day of the Week", "count": "Number of Messages"})

    # Plot 2: Number of messages by month
    fig2 = px.bar(dfs["messages"]["month"].value_counts().sort_index().reset_index(), x="month", y="count", title="Number of Messages by Month", labels={"month": "Month", "count": "Number of Messages"})

    # Plot 3: Number of messages by hour of the day
    fig3 = px.bar(dfs["messages"]["hour"].value_counts().sort_index().reset_index(), x="hour", y="count", title="Number of Messages by Hour of the Day", labels={"hour": "Hour of the Day", "count": "Number of Messages"})

    # Plot 4: Time between messages
    fig4 = px.histogram(dfs["messages"], x="difference", title="Time between Messages", labels={"difference": "Time between Messages (s)", "count": "Number of Messages"})

    # Plot 5: Sentiment Analysis
    fig5 = px.histogram(dfs["messages"], x="sentiment", title="Sentiment Analysis", labels={"sentiment": "Sentiment", "count": "Number of Messages"})

    # Plot 6: Most common words stacked bar chart by person
    fig6 = px.bar(dfs["most_common_words"], x="word", y="count", color="name", title="Most Commonly-Used Words", labels={"word": "Word", "count": "Count", "name": "Person"})

    # Plot 7: Pie chart of total messages by person
    fig7 = px.pie(dfs["stats"].drop(["Total Messages", "Average Messages per Day", "Median difference between messages", "Median difference between replies", "Average message length", "Average sentiment"], axis=1).transpose(), names=dfs["stats"].drop(["Total Messages", "Average Messages per Day", "Median difference between messages", "Median difference between replies", "Average message length", "Average sentiment"], axis=1).columns, values=dfs["stats"].drop(["Total Messages", "Average Messages per Day", "Median difference between messages", "Median difference between replies", "Average message length", "Average sentiment"], axis=1).values[0], title="Total Messages by Person")

    return [fig1, fig2, fig3, fig4, fig5, fig6, fig7]

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



    

    


