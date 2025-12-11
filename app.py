import streamlit as st
import os
import tempfile
import sys
import time
from typing import Tuple, Optional

# --- Gemini SDK Imports ---
try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError as GeminiAPIError
except ImportError:
    st.error("The 'google-genai' library is not installed. Please install it using 'pip install google-genai'.")
    st.stop()


# --- Configuration and Client Initialization ---
# ❗ IMPORTANT: Ensure GEMINI_API_KEY is set in Streamlit secrets
try:
    # Attempt to retrieve API Key from Streamlit secrets
    API_KEY = st.secrets["GEMINI_API_KEY"] 
except KeyError:
    st.error("🚨 API Key Error: Please set 'GEMINI_API_KEY' in your Streamlit secrets file or Streamlit Cloud Secrets.")
    st.stop()

try:
    # Initialize the Gemini Client globally
    client = genai.Client(api_key=API_KEY)  
except Exception as e:
    st.error(f"Error initializing AI client. Details: {e}")
    st.stop()
    
# Model name used for transcription and summarization
# --- CHANGE: Using the faster/free tier model ---
MODEL_NAME = "gemini-2.5-flash" 


# --- Utility Function: Core Logic ---

def analyze_media_with_gemini(uploaded_file, mime_type: str) -> Tuple[Optional[str], str]:
    """
    1. Uploads the audio/video file to the Gemini File API.
    2. Sends the file to the chosen Gemini model for transcription and summarization.
    3. Deletes the file from the File API after use.
    
    Returns: (analysis_result_text, detected_language_code)
    """
    
    st.info(f"Step 1/2: Uploading file **{uploaded_file.name}**")
    
    uploaded_file.seek(0)
    temp_path = None
    
    try:
        # Create a temporary file on disk for the SDK to upload
        file_suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name
        
        # 1. Upload the file to the Gemini File API
        # FIX APPLIED: Removed 'mime_type' keyword argument. The SDK handles detection.
        gemini_file = client.files.upload(file=temp_path)
        st.success(f"File uploaded successfully.")

        # --- Dynamic Prompts ---
        system_instruction = (
            "You are a professional audio/video summarizer. Your task is two-fold: "
            "1. First, create a complete, accurate, and detailed **TRANSCRIPT** of the entire audio content. "
            "2. Second, analyze the transcript and extract the 5 most critical learning points, concepts, or steps discussed, and present them as a bulleted **SUMMARY**. "
            "You MUST output the result in the following structured format, and use the detected language of the media for the SUMMARY (Burmese, English, etc.):"
        )
        
        user_query = (
            "Please analyze the provided file. First, generate the full transcript. "
            "Second, provide a concise summary (5 key points). "
            "Format the output strictly as:\n"
            "## 📝 Full Transcript\n"
            "[The complete, verbatim transcription text here]\n\n"
            "## ✅ Key Point Summary (5 Points)\n"
            "[The 5 key points in bullet-point format, using the primary language of the speech in the audio/video]"
        )

        # 2. Call Gemini for Transcription and Summarization
        st.info(f"Step 2/2: Sending file to AI model for analysis...")
        start_time = time.time()
        
        response = client.models.generate_content(
            model=MODEL_NAME, # Using gemini-2.5-flash
            contents=[user_query, gemini_file], # Pass both the prompt and the file part
            config=types.GenerateContentConfig( 
                system_instruction=system_instruction,
                temperature=0.0 # Keep analysis factual
            )
        )
        
        end_time = time.time()
        st.success(f"Analysis completed in {end_time - start_time:.2f} seconds.")
        
        return response.text, "Unknown"
            
    except GeminiAPIError as e: 
        st.error(f"API Call Failed: {e}")
        return "Analysis failed due to API connection error.", ""
    except Exception as e:
        # This will now catch other errors, including if the API returns an error on file upload
        st.error(f"An unexpected error occurred: {e}")
        return "Analysis failed due to an unexpected error.", ""
    finally:
        # 3. Clean up: Delete the file from the Gemini File API
        if 'gemini_file' in locals():
            st.info(f"Cleaning up: Deleting file from API: {gemini_file.name}")
            try:
                client.files.delete(name=gemini_file.name)
            except Exception as e:
                st.warning(f"Could not delete uploaded file. Please check the files dashboard if necessary. Error: {e}")

        # Clean up the temporary local file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# --- Streamlit UI ---
st.set_page_config(page_title="Video/Audio Summarizer)", layout="centered")

st.markdown("""
<style>
    /* Custom Styling for aesthetics */
    .stButton>button {
        background-color: #0A66C2; 
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
        transition: background-color 0.3s;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background-color: #004182;
    }
    .main-header {
        color: #0A66C2; 
        font-weight: bold;
        text-align: center;
        padding-bottom: 10px;
        border-bottom: 2px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


st.markdown(f'<h1 class="main-header">🎙️Video/Audio Summarizer</h1>', unsafe_allow_html=True)
# --- CHANGE: Updated Model Note ---
st.info("⚡System supports multilingual transcription and summarization.")
st.write("Upload **any** video or audio file (e.g., MP3, MP4, WAV) up to **50MB**. Gemini will automatically transcribe it and provide a key point summary.")

# File Uploader
# File types for the Streamlit file uploader (extensions)
ALL_MEDIA_EXTENSIONS = [
    "mp4", "mov", "wav", "mp3", "m4a", "mkv", "avi", "flv", "wmv", 
    "ogg", "flac", "webm"
]

uploaded_file = st.file_uploader(
    "Upload Video or Audio File",
    type=ALL_MEDIA_EXTENSIONS,
    accept_multiple_files=False
)

if uploaded_file is not None:
    # Determine MIME type 
    mime_type = uploaded_file.type 
    
    # Fallback to determine MIME type from extension if Streamlit's detection is generic
    if not mime_type or 'octet-stream' in mime_type:
        ext = os.path.splitext(uploaded_file.name)[1].lower().replace('.', '')
        if ext == 'mp3': mime_type = 'audio/mpeg'
        elif ext == 'wav': mime_type = 'audio/wav'
        elif ext == 'mp4': mime_type = 'video/mp4'
        elif ext == 'mov': mime_type = 'video/quicktime'
        elif ext == 'm4a': mime_type = 'audio/m4a'
        elif ext == 'ogg': mime_type = 'audio/ogg'
        else: mime_type = 'application/octet-stream' # Default fallback
        
    st.success(f"File uploaded: **{uploaded_file.name}** (Detected MIME: `{mime_type}`) - Ready to process.")
    
    if st.button("Generate Transcript and Summary"):
        
        # Check size limit (Gemini File API limit)
        if uploaded_file.size > (50 * 1024 * 1024): 
            st.error("File size limit exceeded. Please upload a file smaller than 50MB for reliable processing via the File API.")
        else:
            # Main processing function call
            with st.spinner(f"Processing with {MODEL_NAME}..."):
                analysis_result, _ = analyze_media_with_gemini(uploaded_file, mime_type)
            
            # Display the result (which is already formatted with Markdown headings)
            if analysis_result and not analysis_result.startswith("Analysis failed"):
                st.markdown(analysis_result)
                st.success(f"Process complete: Transcription and Summary extracted by {MODEL_NAME}.")
            else:
                st.error("The analysis failed. Please check the error messages above for details.")
