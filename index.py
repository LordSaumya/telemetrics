import streamlit as st
from data import load_data
from analysis import *
import time
import random

def main():
    st.set_page_config(page_title="TeleMetrics", page_icon="📊", layout="wide")
    st.title("TeleMetrics")
    st.sidebar.write("A simple, powerful app that provides insights into your Telegram chat history.")
    st.sidebar.write("Upload your Telegram chat history and get started!\n")
    st.sidebar.write("""
                     To download your chat history,
                     1. Open Telegram Desktop and navigate to the chat you wish to analyse.
                     2. Click on the three dots in the top right corner.
                     3. Select 'Export Chat History'.
                     4. Uncheck all media options (photos, videos, etcetera).
                     5. Change the format to JSON by clicking on the format and selecting 'Machine-readable JSON' in the export format settings, and click 'save'.\n
                     6. Click on the export button and save the file to your computer.
                     """)
    st.sidebar.markdown("__Made with :coffee: by [LordSaumya](https://github.com/LordSaumya)__")

    start_page = st.empty()
    analysis_page = st.empty()

    with start_page.container():
        chat_type = st.radio("Which chat do you want to analyse?", ["Telegram", "Instagram"], index=0, horizontal=True)
        uploaded_file = st.file_uploader("Upload your chat as a JSON file", type=["json"])
        if uploaded_file is not None:
            bytes_data = uploaded_file.read()
            st.success(f"File uploaded successfully: {uploaded_file.name}")
            df = load_data(bytes_data, chat_type)
            if df is None:
                st.error("There was an error loading the data. Please try again.")
            else:
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.write(df)
                with col2:
                    st.info("Is this the file you want to use?")
                    if st.button("Yes, continue"):
                        with st.spinner("Reading your texts..."):
                            time.sleep(random.randint(2, 4))
                        start_page.empty()
                        display_analysis(analysis_page, df)

def display_analysis(analysis_page, df):
    with analysis_page.container():
        st.balloons()
        st.write("Here are some insights into your chat history!")
        dfs = analyse_data(df)
        stats = dfs["stats"].copy()
        for col in stats.columns[6:]:
            stats = stats.rename(columns={col: "Number of messages by " + col})
        stats = stats.rename(columns = {"Median difference between messages": "Median time between messages (sec)", "Median difference between replies": "Median time between replies (sec)", "Average message length": "Average message length (characters)", "Average sentiment": "Average sentiment (between -1 and 1)"})
        stats = stats.transpose().rename(columns={0: "statistic", 1: "value"})
        messages = dfs["messages"].copy()
        most_common_words = dfs["most_common_words"].copy()
        most_common_words = most_common_words.rename(columns={"total_count": "frequency"})

        with st.expander("Statistics", expanded=True):
            col1, col2 = st.columns([0.6, 0.4], gap = "medium")
            with col1:
                st.markdown("**General Statistics**")
                st.dataframe(stats, use_container_width=True)

            with col2:
                st.markdown("**Most Common Words**")
                st.dataframe(most_common_words[["word", "frequency"]].drop_duplicates().reset_index(drop=True), use_container_width=True)
                st.caption("Words under 5 characters are excluded.")
        with st.spinner("Generating graphs..."):
            fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10 = create_graphs(dfs)
        with st.expander("Graphs", expanded=True):
            col1, col2 = st.columns([1, 1], gap = "medium")
            with col1:
                st.plotly_chart(fig1)
                st.plotly_chart(fig3)
                st.plotly_chart(fig5)
                st.plotly_chart(fig7)
                st.plotly_chart(fig9)
            with col2:
                st.plotly_chart(fig2)
                st.plotly_chart(fig4)
                st.plotly_chart(fig6)
                st.plotly_chart(fig8)
                st.plotly_chart(fig10)

        if messages["name"].unique().size == 2:
            st.divider()
            st.subheader("Relationship Score")
            relationship_score, message_score, time_span_score, reply_score, sentiment_score = rel_score(dfs["messages"], dfs["stats"])
            # Relationship score explanation
            if relationship_score < -0.9:
                status = "Mortal enemies"
                colour = "red"
                description = "Epic clashes and fiery showdowns. You hate each other with a passion. You would probably be the first suspects if the other disappeared."
            elif relationship_score < -0.4:
                status = "Don't like each other"
                colour = "orange"
                description = "Passive-aggressive comments and rolling eyes. Best to keep a safe distance from each other."
            elif relationship_score < -0.2:
                status = "Probably neutral"
                colour = "grey"
                description = "You don't really care about each other. Just two people sharing the same planet, barely noticing each other."
            elif relationship_score < 0.2:
                status = "Like each other"
                colour = "green"
                description = "Casual smiles, friendly nods, and the occasional dinner. You're on good terms with each other."
            elif relationship_score < 0.4:
                status = "Good friends"
                colour = "blue"
                description = "Shared adventures and chill conversations. Good times are always on the agenda."
            elif relationship_score < 0.9:
                status = "Best friends"
                colour = "violet"
                description = "Inside jokes and heartfelt conversations. You're always in sync. You're probably the first person they call when something happens."
            else:
                status = "Lovers"
                colour = "rainbow"
                description = "You're the sun to their moon, the peanut butter to their jelly. You're probably already planning your wedding."
            
            st.metric("Relationship score", relationship_score)
            explanation = st.expander(f":{colour}[{status}]")
            explanation.write(description)
            explanation.divider()

            explanation.metric("Message score", message_score)
            if message_score < 0.3:
                explanation.error("You don't really talk to each other.")
            elif message_score < 0.6:
                explanation.info("You talk to each other occasionally.")
            else:
                explanation.success("You talk to each other quite a bit.")
            explanation.caption("The message score (between 0 and 1) is calculated based on the mean message frequency. A lower score means you don't talk much.")
            explanation.divider()

            explanation.metric("Time span score", time_span_score)
            if time_span_score < 0.3:
                explanation.error("You haven't known each other for very long, only a couple months or so.")
            elif time_span_score < 0.6:
                explanation.info("You've known each other for a while. About half a year, maybe more.")
            else:
                explanation.success("You've known each other for a while now. More than a year, maybe even several years.")
            explanation.caption("The time span score (between 0 and 1) is calculated based on when you first started chatting. A lower score means you haven't known each other for very long.")
            explanation.divider()

            explanation.metric("Reply score", reply_score)
            if reply_score < 0.3:
                explanation.error("You don't reply to each other very quickly. You couldn't care less about each other's messages.")
            elif reply_score < 0.6:
                explanation.info("You reply to each other fairly quickly.")
            else:
                explanation.success("You reply to each other very quickly. It's almost as if you're waiting for each other's messages.")
            explanation.caption("The reply score (between 0 and 1) is calculated based on the median time between replies. A higher score means you reply faster.")
            explanation.divider()

            explanation.metric("Sentiment score", sentiment_score)
            if sentiment_score < -0.2:
                explanation.error("Your conversations are mostly negative. You might want to work on that.")
            elif sentiment_score < 0.2:
                explanation.info("Your conversations are mostly neutral. You probably only talk about school or work stuff.")
            else:
                explanation.success("Your conversations are mostly positive. Nice work!")
            explanation.caption("The sentiment score (between -1 and 1) is calculated based on the mean sentiment of your conversations. A lower score means your conversations are more negative.")
            st.caption("The relationship score is an aggregate score based on the message score, time span score, reply score, and sentiment score. -1 means you're mortal enemies, 1 means you're lovers.")
if __name__ == "__main__":
    main()