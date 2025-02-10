import streamlit as st
import tempfile
import os
from transformers import pipeline

# Mapping of display names to Hugging Face model identifiers.
MODEL_MAP = {
    "Whisper Large": "openai/whisper-large",
    "Whisper Base": "openai/whisper-small",
    "Wav2Vec": "facebook/wav2vec2-base-960h",
}


def load_model(model_choice):
    """
    Loads the specified ASR model using Hugging Face's pipeline.
    """
    model_id = MODEL_MAP[model_choice]
    st.info(f"Loading {model_choice} model. This may take a while...")
    # Create a pipeline for automatic speech recognition.
    asr_pipeline = pipeline("automatic-speech-recognition", model=model_id)
    return asr_pipeline


def main():
    st.title("ASR Models Demo")
    st.write(
        """
        Transcribe audio using one of the following models:

        - **Whisper Large**
        - **Whisper Base**
        - **Wav2Vec**

        The model is loaded on demand, so only one model is active at a time.
        """
    )

    # Sidebar: model selection.
    model_choice = st.sidebar.selectbox("Select ASR Model", list(MODEL_MAP.keys()))

    # Load the selected model only if it's not already loaded.
    if "current_model" not in st.session_state or st.session_state.current_model != model_choice:
        st.session_state.model_pipeline = load_model(model_choice)
        st.session_state.current_model = model_choice

    # Audio file uploader.
    audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "m4a"])

    if st.button("Transcribe"):
        if audio_file is None:
            st.error("Please upload an audio file.")
            return

        # Save the uploaded audio file to a temporary file.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        model_pipeline = st.session_state.model_pipeline

        try:
            # For Whisper models, you can optionally request word-level timestamps.
            if model_choice in ["Whisper Large", "Whisper Base"]:
                result = model_pipeline(tmp_path, return_timestamps="word")
            else:
                result = model_pipeline(tmp_path)
        except Exception as e:
            st.error(f"Error during transcription: {e}")
            result = None

        # Remove the temporary file.
        os.remove(tmp_path)

        if result is not None:
            st.markdown("### Transcription:")
            st.write(result)


if __name__ == "__main__":
    main()
