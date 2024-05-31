import streamlit as st
from data import load_data
from analysis import *

def main():
    st.set_page_config(page_title="TeleMetrics", page_icon="📊")
    st.title("TeleMetrics")
    st.sidebar.write("A simple yet powerful app that provides insights into your Telegram chat history.")
    st.sidebar.write("Upload your Telegram chat history and get started!")

    start_page = st.empty()
    analysis_page = st.empty()

    with start_page.container():
        uploaded_file = st.file_uploader("Upload your Telegram chat as a JSON file", type=["json"])
        if uploaded_file is not None:
            bytes_data = uploaded_file.read()
            st.success(f"File uploaded successfully: {uploaded_file.name}")
            df = load_data(bytes_data)
            if df is None:
                st.error("There was an error loading the data. Please try again.")
            else:
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.write(df)
                with col2:
                    st.info("Is this the file you want to use?")
                    if st.button("Yes, continue"):
                        start_page.empty()
                        display_analysis(analysis_page, df)

def display_analysis(analysis_page, df):
    with analysis_page.container():
        st.write("Here are some insights into your chat history:")
        dfs = analyse_data(df)
        stats = dfs["stats"]
        messages = dfs["messages"]
        most_common_words = dfs["most_common_words"]

        st.write("Statistics")
        st.write(stats)

        st.write("Messages")
        st.write(messages)

        st.write("Most common words")
        st.write(most_common_words)

        if messages["name"].unique().size == 2:
            st.write("Your relationship score is ", rel_score(messages, stats))

        st.write("Graphs")
        graphs = create_graphs(dfs)
        for graph in graphs:
            st.plotly_chart(graph)

if __name__ == "__main__":
    main()