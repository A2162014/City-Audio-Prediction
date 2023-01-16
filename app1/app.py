import librosa
import numpy as np
import streamlit as st
import tensorflow as tf

try:
    st.title("City Audio Predictions")
    st.write("This model predicts the type of an audio file based on 10 common city sounds, they are as follows:")
    st.markdown("- air condition")
    st.markdown("- car horn")
    st.markdown("- children playing")
    st.markdown("- dog bark")
    st.markdown("- drilling")
    st.markdown("- engine idling")
    st.markdown("- gun shot")
    st.markdown("- jackhammer")
    st.markdown("- siren")
    st.markdown("- street music")


    def aud_class(file):
        # preprocess the audio file
        audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=8732)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

        # Reshape MFCC feature to 2-D array
        mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

        new_model = tf.keras.models.load_model('my_model.h5')

        # predicted_label=model.predict_classes(mfccs_scaled_features)
        x_predict = new_model.predict(mfccs_scaled_features)
        predicted_label = np.argmax(x_predict, axis=1)

        # 0 = air_conditioner
        # 1 = car_horn
        # 2 = children_playing
        # 3 = dog_bark
        # 4 = drilling
        # 5 = engine_idling
        # 6 = gun_shot
        # 7 = jackhammer
        # 8 = siren
        # 9 = street_music

        if predicted_label == [0]:
            st.write('The audio contains sound of air condition')
        elif predicted_label == [1]:
            st.write('The audio contains sound of car horn')
        elif predicted_label == [2]:
            st.write('The audio contains sound of children playing')
        elif predicted_label == [3]:
            st.write('The audio contains sound of dog bark')
        elif predicted_label == [4]:
            st.write('The audio contains sound of drilling')
        elif predicted_label == [5]:
            st.write('The audio contains sound of engine idling')
        elif predicted_label == [6]:
            st.write('The audio contains sound of gun shot')
        elif predicted_label == [7]:
            st.write('The audio contains sound of jackhammer')
        elif predicted_label == [8]:
            st.write('The audio contains sound of siren')
        else:
            st.write('The audio contains sound of street music')

        audio_file = open(file, 'rb')
        audio_bytes = audio_file.read()

        st.audio(audio_bytes, format='audio/ogg')

        st.write("*app and model is prepared by Ashvath S.P*")


    file_path = st.text_input('Enter an absolute file path of an audio file')

    aud_class(file_path)
except:
    # Prevent the error from propagating into your Streamlit app.
    pass
