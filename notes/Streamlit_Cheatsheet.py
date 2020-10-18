import time
import datetime
import streamlit as st

# To run :- streamlit run Streamlit_Cheatsheet.py

st.title("StreamLit Tutorial")

st.header("This is header smaller font than title")
st.subheader("This is a subheader")

st.text("Smallest test")

st.markdown("### This is a Markdown")

st.success("Similar to Bootstrap success alert")
st.info("Similar to Bootstrap info alert")
st.warning("Similar to Bootstrap warning alert yellow")
st.error("Similar to Bootstrap error alert red")

st.exception("NameError('name three not defined')")

# Get help about python function
st.help(range)

# Writing Text
st.write("Text with write")
st.write(range(10))

# Showing Images
# from PIL import Image
# img = Image.open("1.jpg")
# st.image(img , width = 300 , caption = "Simple Image")

# Showing Video
# vid_file = open("example.mp4","rd").read()
# st.video(vid_file)

# Adding Audio file
# audio_file =  open("file.mp3","rb").read()
# st.audio(audio_file, format = "audio/mp3")

# Widgets

# Cjeckbox
if st.checkbox("Show/Hide"):
    st.text("Showing Text")

# Radio
status = st.radio("What is your status", ("Active", "Inactive"))
if status == "Active":
    st.success("You are Active")
else:
    st.warning("You are Inactive")

# Select Box
occupation = st.selectbox(
    "Your Occupation", ["Programmer", "DataScientist", "Doctor", "Police"])
st.write("You Selected this option ", occupation)

# MultiSelect
location = st.multiselect("Where do you work ? ",
                          ("London", "Mumbai", "New York", "Delhi"))
st.write("You Selected ", len(location), "locations")

# Slider
level = st.slider("What's is your level", 1, 5)

# Buttons
st.button("Simple Button")

if st.button("About"):
    st.text("About button activated")

# Text Input
firstname = st.text_input("Enter your first name", "Type here ...")
if st.button("Submit"):
    result = firstname.title()
    st.success(result)

# Text Area
message = st.text_area("Enter your meassage", "Type here ...")
# if st.button("Submit"):
#      result = message.title()
#      st.success(result)

# Date Input
today = st.date_input("Today is", datetime.datetime.now())
# Time
time = st.time_input("The time is ", datetime.time())

# Dislaying JSON
st.text("Displaying JSON")
st.json({"name": 'Abhi', "age": 20, "gender": "male"})

# Displaying Raw Code
st.text("Displaying Raw Code")
st.code("import numpy as np")

# or

with st.echo():
    import pandas as pd
    df = pd.DataFrame()

# Progress Bar
my_bar = st.progress(0)
for p in range(10):
    my_bar.progress(p + 1)

# Spinner
with st.spinner("Waiting ... "):
    time.sleep(5)
st.success("Finished")

# Ballons
st.balloons()

# Sidebar
st.sidebar.header("About")
st.sidebar.text("small text")

# Functions


@st.cache  # to make it faster
def run_fxn():
    return range(100)


st.write(run_fxn())

# Plots
st.pyplot()

# Dataframe
st.dataframe(df)

# Tables
st.table(df)
