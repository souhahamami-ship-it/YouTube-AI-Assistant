import streamlit as st

import youtube_helper as yh
import summarizer
import langchain_helper as lch


# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="YouTube AI Assistant",
    page_icon="🎥",
    layout="wide"
)


# -----------------------------------
# LOAD HUGGING FACE MODEL
# -----------------------------------

@st.cache_resource
def load_summarization_model():

    return summarizer.load_model()


# -----------------------------------
# LOAD VECTOR DATABASE
# -----------------------------------

@st.cache_resource
def create_vector_database(transcript):

    return lch.create_vectordb_from_transcript(
        transcript
    )


# -----------------------------------
# SESSION STATE
# -----------------------------------

if "transcript" not in st.session_state:
    st.session_state.transcript = None

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "summary" not in st.session_state:
    st.session_state.summary = None

if "processed_url" not in st.session_state:
    st.session_state.processed_url = None


# -----------------------------------
# TITLE
# -----------------------------------

st.title("🎥 YouTube AI Assistant")

st.write(
    "Summarize YouTube videos and ask questions "
    "about their content using AI."
)


# -----------------------------------
# PROCESS VIDEO
# -----------------------------------

youtube_url = st.text_input(
    "Paste a YouTube URL"
)


if st.button("Process Video"):

    if not youtube_url:

        st.warning(
            "Please enter a YouTube URL."
        )

    else:

        try:

            # Get transcript
            with st.spinner(
                "Getting video transcript..."
            ):

                transcript = (
                    yh.get_transcript_from_url(
                        youtube_url
                    )
                )

                st.session_state.transcript = (
                    transcript
                )

                st.session_state.processed_url = (
                    youtube_url
                )

                # Reset old results
                st.session_state.summary = None


            # Create vector database
            with st.spinner(
                "Preparing video for questions..."
            ):

                db = create_vector_database(
                    transcript
                )

                st.session_state.vector_db = db


            st.success(
                "Video processed successfully!"
            )

            st.info(
                f"Transcript length: "
                f"{len(transcript)} characters"
            )

        except Exception as e:

            st.error(
                f"An error occurred: {e}"
            )


# -----------------------------------
# SHOW FEATURES ONLY AFTER PROCESSING
# -----------------------------------

if st.session_state.transcript:

    tab1, tab2 = st.tabs([
        "📝 Summarize Video",
        "💬 Ask Questions"
    ])


    # =================================
    # SUMMARY TAB
    # =================================

    with tab1:

        st.subheader("📝 Video Summary")

        if st.button("Generate Summary"):

            try:

                with st.spinner(
                    "Loading AI model..."
                ):

                    tokenizer, model = (
                        load_summarization_model()
                    )


                progress_bar = st.progress(0)

                with st.spinner(
                    "Summarizing video..."
                ):

                    def update_progress(value):

                        progress_bar.progress(value)


                    final_summary, num_chunks = (
                        summarizer.summarize_transcript(
                            st.session_state.transcript,
                            tokenizer,
                            model,
                            update_progress
                        )
                    )

                    st.session_state.summary = (
                        final_summary
                    )


                st.success(
                    "Summary created!"
                )

                st.caption(
                    f"Processed using "
                    f"{num_chunks} transcript chunks."
                )

            except Exception as e:

                st.error(
                    f"An error occurred: {e}"
                )


        if st.session_state.summary:

            st.write(
                st.session_state.summary
            )


    # =================================
    # QUESTION AND ANSWER TAB
    # =================================

    with tab2:

        st.subheader(
            "💬 Ask About the Video"
        )

        query = st.text_area(
            "Ask a question about the video"
        )

        groq_api_key = st.text_input(
            "Groq API Key",
            type="password"
        )


        if st.button("Ask Question"):

            if not query:

                st.warning(
                    "Please enter a question."
                )

            elif not groq_api_key:

                st.warning(
                    "Please enter your Groq API key."
                )

            else:

                try:

                    with st.spinner(
                        "Finding the answer..."
                    ):

                        response = (
                            lch.get_response_from_query(
                                st.session_state.vector_db,
                                groq_api_key,
                                query
                            )
                        )


                    st.subheader("Answer")

                    st.write(response)


                except Exception as e:

                    st.error(
                        f"An error occurred: {e}"
                    )