import streamlit as st 

#configure the application 
st.set_page_config(
    page_title="DocuMind",
    page_icon="📄",
    layout="wide"
)

#application heading 
st.title("📄 DocuMind")
st.subheader(" Your AI Document Assistant")

st.write("Upload your PDF and ask about its content")



#pdf upload 

uploaded_file = st.file_uploader("Upload your PDF" , type=["pdf"])


question_input = st.text_input(" What would you like to know about your documents ")



#submit button 

if st.button("Ask DocuMind"):
    if uploaded_file is None:
        st.warning("Please uplaod a pdf ")
    elif  not question_input.strip():
        st.warning("Please enter a question")
    else: 
        st.info("AI functionality coming soon ")

