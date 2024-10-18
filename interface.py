import streamlit as st
#from streamlit_chat import message as st_message
from audiorecorder import audiorecorder
from st_audiorec import st_audiorec
import speech_recognition as sr
from melo.api import TTS
import tempfile
import shutil
import requests
import json
import base64
from translate import Translator




#Voice recognition model
def recognize_speech(audio_file):
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = r.record(source)  # Ses dosyasını oku
    text = r.recognize_google(audio_data, language='en-US') 
    return text
    """#try:
        text = r.recognize_google(audio_data, language='en-US')
        return text
    except sr.UnknownValueError:
        return "Sorry, I could not understand the audio."
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"
        """




#def generate_audio()


url = "http://127.0.0.1:8000/get_question_from_messages/"

def generate_answer(audio):

    """
    INPUT: QUESTIONS OF TOURIST BY VOICE RECORDING
    
    OUTPUT: INFORMATION OF MACCHU PICCHU
    """

    with st.spinner("Hanna replies ..."):

        
        # To save audio to a file:
        #file = open('audio.wav', 'w')
        #temp_file_path = ""


        
        audio.export("./audio.wav", format="wav")
                
        # Voice recognition model
        
        text = recognize_speech("./audio.wav")
        #text = recognize_speech(desired_file_name)

        #Response question
        #chain = get_model()

        st.session_state.messages.append({"role": "user", "content": text})
        #try:
        data = {
            "content":st.session_state.messages
        }
        json_data = json.dumps(data)
        #answer_guide = chain.run(text)
        #print(st.session_state.messages)
        stream = requests.post(url, data=json_data, headers={'Content-Type': 'application/json'})
        st.session_state.messages = stream.json()
        answer_guide = st.session_state.messages[-1]['content']

        #except:

        #    answer_guide = "Sorry. I don't understand"

        #Save conversation
        #st.session_state.history.append({"message": text, "is_user": True})
        #st.session_state.history.append({"message": f"{answer_guide}", "is_user": False})


        #st.success("Question Answered")

    return answer_guide

if __name__ == "__main__":

    # Remove the hamburger in the upper right hand corner and the Made with Streamlit footer
    hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
        """
    st.markdown(hide_menu_style, unsafe_allow_html=True)
    
    # Store the conversation in the GUI

    if "model" not in st.session_state:
    
        st.session_state.speed = 0.8
        
        # CPU is sufficient for real-time inference.
        # You can set it manually to 'cpu' or 'cuda' or 'cuda:0' or 'mps'
        device = 'auto' # Will automatically use GPU if available
        
        # English 
        #text = "Did you ever hear a folk tale about a giant turtle?"
        st.session_state.model = TTS(language='EN', device=device)
        st.session_state.speaker_ids = st.session_state.model.hps.data.spk2id
        
        # American accent
        st.session_state.output_path = 'speech.wav'
        #st.session_state.model.tts_to_file(text, speaker_ids['EN-US'], output_path, speed=speed)
        


    if "translator" not in st.session_state:
        st.session_state["translator"] = Translator(to_lang="tr")
        


    
    if "messages" not in st.session_state:

        st.session_state.messages = []

    if "audio_content" not in st.session_state:
        st.session_state.audio_content = {"assistant": [],"user":[]}

    # Image
    st.image("./teacher.png")

 
    # Title
    st.title("Conversation with a English Teacher")

              
    #Audio Recorder Button
    #audio = st_audiorec()
    audio = audiorecorder("Say something", "Recording")
    if len(audio) > 0:
        st.session_state.audio_content['user'].append(audio)

    

     
        # Answer of the model
        answer = generate_answer(audio)
        turkish_answer = st.session_state["translator"].translate(answer)
        print("***"*50)
        print(turkish_answer)
        print("***"*50)
        st.session_state.model.tts_to_file(answer,st.session_state.speaker_ids['EN-US'],"answer.wav", speed=st.session_state.speed)
        with open("answer.wav", "rb") as f:

            answer_audio = f.read()
            audio_base64 = base64.b64encode(answer_audio).decode('utf-8')
        #with open(audio_file, "rb") as f:
        #    data = f.read()
        st.session_state.audio_content["assistant"].append(answer_audio)
        
        
        
        #Show historical conversation
        #audio_data = st.session_state["audio_content"]["assistant"][-1]

        #audio_base64 = base64.b64encode(answer_audio).decode('utf-8')
        #audio_base64 = base64.b64encode(answer_audio).decode('utf-8')

        st.audio(st.session_state["audio_content"]["assistant"][-1], format='audio/wav',autoplay=True)#@st.cache_resource()
        #for i, chat in enumerate(st.session_state.audio_content['assistant']): 

         #   st.audio(st.session_state["audio_content"]["assistant"][i], format='audio/wav')#@st.cache_resource()
            












