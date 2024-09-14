import streamlit as st
import pandas as pd
import pickle
import time
import numpy as np
import zipfile
import os
import langchain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.document_loaders import DirectoryLoader
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
import time
import torch
import numpy as np
import cv2
from PIL import Image
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle
import os

# Set page configuration
st.set_page_config(
    page_icon="🇸",
    layout="wide",
    initial_sidebar_state="expanded",
)

#SENTIMENTAL ANALYSIS
vader_lexicon_path = os.path.expanduser("/Users/shanthakumark/Downloads/vader_lexicon.txt")

@st.cache_resource
def load_sentiment_analyzer():
    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.data.load(vader_lexicon_path)  # Loading from local file
    return SentimentIntensityAnalyzer()

# Analyze the sentiment of the text
def analyze_sentiment(text):
    sia = load_sentiment_analyzer()
    sentiment_scores = sia.polarity_scores(text)
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def main():
    st.title("Mobile Review Sentiment Analysis")

    # Input box for user review
    user_input = st.text_input("Enter your mobile review:")

    if user_input:
        sentiment = analyze_sentiment(user_input)
        if sentiment == "Positive":
            with st.container(height = 1000):
                st.header("Customer have a Positive impression on this product",divider = "rainbow")
                st.image("/Users/shanthakumark/Downloads/happy-mochi.gif",use_column_width=False)
                t1,t2,t3 = st.columns([1,1,1])
                with t1:
                    st.markdown(
                        """
                        <h2>Emotion Rating: ⭐️⭐️⭐️⭐️⭐️</h2>
                        <hr style="border: 10px solid green;">
                        """, unsafe_allow_html=True
                            )
        if sentiment == "Negative":
            with st.container(height = 600):
                st.header("Customer have a Negative impression on this product",divider = "rainbow")
                st.image("/Users/shanthakumark/Downloads/so-sad-sad-face.gif",use_column_width=False,width = 300)
                t1_,t2_,t3_ = st.columns([1,1,1])
                with t1_:
                    st.markdown(
                        """
                        <h2>Emotion Rating: ⭐️</h2>
                        <hr style="border: 10px solid red;">
                        """, unsafe_allow_html=True
                            )
        if sentiment == "Neutral":
            with st.container(height = 600):
                st.header("Customer have a Neutral impression on this product",divider = "rainbow")
                st.image("/Users/shanthakumark/Downloads/emm-thinking.gif",use_column_width=False,width = 300)
                t1_1,t2_1,t3_1 = st.columns([1,1,1])
                with t1_1:
                    st.markdown(
                        """
                        <h2>Emotion Rating: ⭐️⭐️⭐️</h2>
                        <hr style="border: 10px solid orange;">
                        """, unsafe_allow_html=True
                            )

# @st.cache_resource
# def load_documents_from_directory():
#     # Load PDF files from directory
#     loader = DirectoryLoader('/Users/shanthakumark/Desktop/Sharing/Final_project/uploaded files', glob="*.pdf", loader_cls=PyPDFLoader)
#     return loader.load()

# @st.cache_resource
# def setup_embedding_and_faiss():
#     # Load documents and process the text
#     documents = load_documents_from_directory()

#     # Split text for more efficient retrieval
#     split_text = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     splitted_text = split_text.split_documents(documents)

#     # Use a more lightweight model for HuggingFace embeddings
#     hf_model_path = "sentence-transformers/paraphrase-MiniLM-L3-v2"
#     embedding = HuggingFaceEmbeddings(model_name=hf_model_path, model_kwargs={"device": "cpu"})

#     # Create FAISS index
#     vector = FAISS.from_documents(splitted_text, embedding=embedding)
#     vector.save_local("/Users/shanthakumark/Desktop/Sharing/Final_project/vector_files")

#     return vector, embedding


# @st.cache_resource
# def load_llm_model():
#     # Load LLaMA model with reduced token limit for fast responses
#     model_path = "/Users/shanthakumark/Downloads/llama-2-7b-chat.ggmlv3.q8_0.bin"
#     return CTransformers(model=model_path, model_type="llama", config={"temperature": 0.5, "max_new_tokens": 50})


# @st.cache_resource
# def load_faiss_index():
#     # Load embeddings
#     hf_model_path = "sentence-transformers/paraphrase-MiniLM-L3-v2"
#     embedding = HuggingFaceEmbeddings(model_name=hf_model_path, model_kwargs={"device": "cpu"})

#     # Load pre-saved FAISS index for faster retrieval
#     vector = FAISS.load_local("/Users/shanthakumark/Desktop/Sharing/Final_project/vector_files", embedding, allow_dangerous_deserialization=True)
#     return vector


# @st.cache_resource
# def cre_prompt():
#     # Custom prompt template
#     prompt = """
#     provide only useful information /
#     give the exact answer to the user /
#     if you do not know the answer, do not try to make over /
#     Context: {context}
#     Question: {question}
#     """
#     prompt_template = PromptTemplate(template=prompt, input_variables=['context', 'question'])
#     return prompt_template

# @st.cache_resource
# def retr_answer(query):
#     # Load LLaMA model and FAISS index
#     runn = setup_embedding_and_faiss()
#     llm_model = load_llm_model()
#     vector = load_faiss_index()

#     # Create the prompt template
#     prompt_template = cre_prompt()

#     # Create the retrieval QA system
#     answer_engine = RetrievalQA.from_chain_type(
#         llm=llm_model,
#         chain_type="stuff",
#         chain_type_kwargs={"prompt": prompt_template, "verbose": False},
#         return_source_documents=False,
#         retriever=vector.as_retriever(search_kwargs={'k': 2})  # Fetch 2 documents for context
#     )

#     # Retrieve and return the answer
#     answer = answer_engine({"query": query})
#     return answer


@st.cache_resource
def load_documents_from_directory():
    # Load PDF files from directory
    loader = DirectoryLoader('/Users/shanthakumark/Desktop/Sharing/Final_project/uploaded files', glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


@st.cache_resource
def setup_embedding_and_faiss():
    # Load documents and create FAISS index only once
    documents = load_documents_from_directory()

    # Split text for efficient retrieval
    split_text = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splitted_text = split_text.split_documents(documents)

    # Load a lightweight HuggingFace embedding model
    hf_model_path = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    embedding = HuggingFaceEmbeddings(model_name=hf_model_path, model_kwargs={"device": "cpu"})

    # Create FAISS index
    vector = FAISS.from_documents(splitted_text, embedding=embedding)
    vector.save_local("/Users/shanthakumark/Desktop/Sharing/Final_project/vector_files")

    return vector


@st.cache_resource
def load_faiss_index():
    # Load pre-saved FAISS index and embedding model for fast retrieval
    hf_model_path = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    embedding = HuggingFaceEmbeddings(model_name=hf_model_path, model_kwargs={"device": "cpu"})

    vector = FAISS.load_local("/Users/shanthakumark/Desktop/Sharing/Final_project/vector_files", embedding, allow_dangerous_deserialization=True)
    return vector


@st.cache_resource
def load_llm_model():
    # Load LLaMA model with reduced token limit for faster responses
    model_path = "/Users/shanthakumark/Downloads/llama-2-7b-chat.ggmlv3.q8_0.bin"
    return CTransformers(model=model_path, model_type="llama", config={"temperature": 0.5, "max_new_tokens": 50})


@st.cache_resource
def cre_prompt():
    # Simplified prompt template to ensure focused responses
    prompt = """
    Provide only the exact answer to the user's question.
    Context: {context}
    Question: {question}
    Answer:
    """
    prompt_template = PromptTemplate(template=prompt, input_variables=['context', 'question'])
    return prompt_template


def retr_answer(query):
    # Load LLaMA model and FAISS index only once
    llm_model = load_llm_model()
    vector = load_faiss_index()

    # Create the prompt template
    prompt_template = cre_prompt()

    # Retrieve relevant documents from FAISS (limit number of documents for faster results)
    docs = vector.similarity_search(query, k=2)

    # Use only the most relevant portion of the retrieved documents for context
    context = "\n".join([doc.page_content for doc in docs])

    # Create final input prompt for the LLaMA model
    final_prompt = prompt_template.format(context=context, question=query)

    # Generate the answer from the LLaMA model
    result = llm_model(final_prompt)

    # Clean up the result to extract only the answer part
    clean_answer = result.strip().split('\n')[0]  # Extract the first line for simplicity

    return clean_answer

#Object Detection Code
yolov5_repo_path = "/Users/shanthakumark/Downloads/yolov5"  # Path to directory containing hubconf.py
sys.path.append(yolov5_repo_path)

# Load YOLOv5 model
@st.cache_resource
def load_model():
    model_path = "/Users/shanthakumark/Downloads/yolov5n.pt"  # Path to yolov5n.pt
    model = torch.hub.load(yolov5_repo_path, 'custom', path=model_path, source='local')
    return model

# Function to perform human detection
def detect_humans(image, model):
    # Convert PIL image to numpy array
    img_np = np.array(image.convert('RGB'))

    # Run YOLOv5 model inference
    results = model(img_np)

    # Extract detections for humans (class 0 is 'person' in COCO dataset)
    human_results = results.pandas().xyxy[0]
    human_results = human_results[human_results['class'] == 0]  # Filter only 'person' class

    # Draw bounding boxes only for humans
    for _, row in human_results.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        img_np = cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img_np, human_results.shape[0]  # Return processed image and human count


@st.cache_resource
def load_all():
    with open("/Users/shanthakumark/Desktop/Sharing/Final_project/pickle_files/encoder_vs.pkl","rb") as encoder_:
        encoder_e = pickle.load(encoder_)
    
    with open("/Users/shanthakumark/Desktop/Sharing/Final_project/pickle_files/model_built_vs.pkl","rb") as model_f:
        model_final = pickle.load(model_f)
    
    with open("/Users/shanthakumark/Desktop/Sharing/Final_project/pickle_files/scaling_vs.pkl","rb") as scaling_:
        scaling_s = pickle.load(scaling_)

    with open("/Users/shanthakumark/Desktop/Sharing/Final_project/pickle_files/Best2.pkl","rb") as best_param:
        best_param = pickle.load(best_param)

    actual_data = pd.read_csv("/Users/shanthakumark/Desktop/Sharing/Final_project/cleaned_up_data/data_cleaned.csv")

    with open("/Users/shanthakumark/Desktop/Sharing/Final_project/pickle_files/first_copy.pkl","rb") as w_o_encoding:
        without_en = pickle.load(w_o_encoding)
    
    with open("/Users/shanthakumark/Desktop/Sharing/Final_project/pickle_files/f_data.pkl","rb") as a_encoding:
        w_encoding = pickle.load(a_encoding)


    with open("/Users/shanthakumark/Desktop/Sharing/Final_project/pickle_files/gender_coded.pkl","rb") as gender_:
        gender = pickle.load(gender_)

    with open("/Users/shanthakumark/Desktop/Sharing/Final_project/pickle_files/name_contract.pkl","rb") as name_contract_:
        name_contract = pickle.load(name_contract_)

    with open("/Users/shanthakumark/Desktop/Sharing/Final_project/pickle_files/NAME_EDUCATION_TYPE.pkl","rb") as NAME_EDUCATION_TYPE_:
        NAME_EDUCATION_TYPE =pickle.load(NAME_EDUCATION_TYPE_)

    with open("/Users/shanthakumark/Desktop/Sharing/Final_project/pickle_files/NAME_HOUSING_TYPE.pkl","rb") as NAME_HOUSING_TYPE_:
        NAME_HOUSING_TYPE = pickle.load(NAME_HOUSING_TYPE_)

    with open("/Users/shanthakumark/Desktop/Sharing/Final_project/pickle_files/NAME_INCOME_TYPE.pkl","rb") as NAME_INCOME_TYPE_:
        NAME_INCOME_TYPE = pickle.load(NAME_INCOME_TYPE_)

    with open("/Users/shanthakumark/Desktop/Sharing/Final_project/pickle_files/NAME_PAYMENT_TYPE_.pkl","rb") as NAME_PAYMENT_TYPE_:
        NAME_PAYMENT_TYPE = pickle.load(NAME_PAYMENT_TYPE_)

    with open("/Users/shanthakumark/Desktop/Sharing/Final_project/pickle_files/OCCUPATION_TYPE_.pkl","rb") as OCCUPATION_TYPE_:
        OCCUPATION_TYPE = pickle.load(OCCUPATION_TYPE_)

    with open("/Users/shanthakumark/Desktop/Sharing/Final_project/pickle_files/WALLSMATERIAL_MODE_.pkl","rb") as WALLSMATERIAL_MODE_:
        WALLSMATERIAL_MODE = pickle.load(WALLSMATERIAL_MODE_)

    with open("/Users/shanthakumark/Desktop/Sharing/Final_project/pickle_files/WEEKDAY_APPR_PROCESS_START_x_.pkl","rb") as WEEKDAY_APPR_PROCESS_START_x_:
        WEEKDAY_APPR_PROCESS_START_x__ = pickle.load(WEEKDAY_APPR_PROCESS_START_x_)

    with open("/Users/shanthakumark/Desktop/Sharing/Final_project/pickle_files/WEEKDAY_APPR_PROCESS_START_x_best2.pkl","rb") as week_day:
        week_day_dic = pickle.load(week_day)

    with open("/Users/shanthakumark/Desktop/Sharing/Final_project/pickle_files/ORGANIZATION_TYPE_best2.pkl","rb") as org_type_best2:
        org_type_best2_dic = pickle.load(org_type_best2)

    with open("/Users/shanthakumark/Desktop/Sharing/Final_project/pickle_files/OCCUPATION_TYPE_best2.pkl","rb") as OCCUPATION_TYPE_best2:
        OCCUPATION_TYPE_best2 = pickle.load(OCCUPATION_TYPE_best2)
        
    with open("/Users/shanthakumark/Desktop/Sharing/Final_project/pickle_files/NAME_FAMILY_STATUS_best2.pkl","rb") as NAME_FAMILY_STATUS_best2:
        NAME_FAMILY_STATUS_best2 = pickle.load(NAME_FAMILY_STATUS_best2)

    return encoder_e,scaling_s,best_param,actual_data,without_en,w_encoding,gender,name_contract,NAME_EDUCATION_TYPE,NAME_HOUSING_TYPE,NAME_INCOME_TYPE,NAME_PAYMENT_TYPE,OCCUPATION_TYPE,WALLSMATERIAL_MODE,WEEKDAY_APPR_PROCESS_START_x__,week_day_dic,org_type_best2_dic,OCCUPATION_TYPE_best2,NAME_FAMILY_STATUS_best2,model_final



encoder_e,scaling_s,best_param,actual_data,without_en,w_encoding,gender_,name_contract_,NAME_EDUCATION_TYPE_,NAME_HOUSING_TYPE_,NAME_INCOME_TYPE_,NAME_PAYMENT_TYPE_,OCCUPATION_TYPE_,WALLSMATERIAL_MODE_,WEEKDAY_APPR_PROCESS_START_x___,week_day_dic,org_type_best2_dic,OCCUPATION_TYPE_best2,NAME_FAMILY_STATUS_best2,model_final_ = load_all()

# encoder_e,scaling_s,best_param,actual_data,without_en,w_encoding,gender_,name_contract_,NAME_EDUCATION_TYPE_,
# NAME_HOUSING_TYPE_,NAME_INCOME_TYPE_,NAME_PAYMENT_TYPE_,OCCUPATION_TYPE_,WALLSMATERIAL_MODE_,WEEKDAY_APPR_PROCESS_START_x_

tab1,tab3,tab4 = st.tabs(["Loan Data Prediction","Object Detection","Sentimental Analysis"])


with tab1:
    # with st.expander(label=" CHECK YOU LOAN ELIGIBILITY - Get upto 3,00,000 Instant Loan"):
        col1,col2,col3 = st.columns([0.2,0.3,0.3])
        with col1:
            st.write(" ")
            st.write(" ")
            st.markdown(f"<h3 style='font-size:32px; color:#DC143C;'>USER DETAILS</h3>", unsafe_allow_html=True)
            

            

            with col2:
                st.write(" ")
                with st.container(height=400):
                    #important for model
                    name_of_cus = st.text_input("Enter Your Name (As per PAN)")
                    ORGANIZATION_TYPE_col12 = st.selectbox(label = "Organisation Type",options=without_en[without_en[best_param["COL"]].select_dtypes(["object"]).columns[0]].unique())
                    OCCUPATION_TYPE_col14 = st.selectbox(label = "Occupation type",options= without_en[without_en[best_param["COL"]].select_dtypes(["object"]).columns[2]].unique())
                    NAME_FAMILY_STATUS_col16 = st.selectbox(label="Current Family Status",options= without_en[without_en[best_param["COL"]].select_dtypes(["object"]).columns[3]].unique())
                
            with col3:
                st.write("")
                with st.container(height=400):
                    AMT_ANNUITY_x_col4 = st.number_input(label = "Enter your annual income",max_value = 61123.5,min_value=1615.5)
                    EXT_SOURCE_1_col5 = st.number_input(label="Select score for External Source Doc - 1",max_value=0.96,min_value=0.014)
                    AMT_CREDIT_x_col7 = st.number_input(label="Final credit amount on the previous application",max_value=1614960.0,min_value=45000.0)
                    AMT_INCOME_TOTAL_col8 = st.number_input(label="Income of the client",min_value=25650.0,max_value=348750.0)

        with col1:
            if name_of_cus:
                with st.container(height=400):
                    st.markdown(f"<h3 style='font-size:24px; color:#ff6347;'>Hey {name_of_cus}....</h3>", unsafe_allow_html=True)
                    st.write(f"Its amazing to know that you're working in '{ORGANIZATION_TYPE_col12}' organisation ......")
                    if NAME_FAMILY_STATUS_col16 == "Single / not married":
                        st.markdown(f"<h3 style='font-size:12px; color:#DA70D6;'>Sorry😞... Soon You'll Get your Life Partner....</h3>", unsafe_allow_html=True)
                    st.image("/Users/shanthakumark/Downloads/happy-animated-cactus-hi-gesture-vlzixxpzh76wgbbh.gif")
        if name_of_cus:
            st.markdown(f"<h3 style = 'font-size:20px; color:#FF1493;'> Hey {name_of_cus}, Relax for a while until our agent fill rest of your details </h3>",unsafe_allow_html=True)                
        
        st.divider()

        colq1,colq2,colq3 = st.columns([0.2,0.2,0.2])
    
            
                


        with colq1:
            with st.container(height=300):
                st.markdown(f"<h3 style='font-size:30px; color:#40E0D0;'>AGENT LOGIN</h3>", unsafe_allow_html=True)
                u_name = st.text_input(label="User Name",max_chars=7)
                pass_ = st.text_input(label = "Enter Your Password",max_chars= 12)
                if u_name == "Shanth" and pass_ == "Shashi@007":
                    print("Enter Customer's details for below mentioned questions")
                else:
                    print("Sorry You've Entered Wrong Password")

        with colq2:
                if u_name == "Shanth" and pass_ == "Shashi@007":
                    with st.container(height=600):
                        WEEKDAY_APPR_PROCESS_START_x_col13 = st.selectbox(label = "On which day of the week did the client apply for previous application ?",options = without_en[without_en[best_param["COL"]].select_dtypes(["object"]).columns[1]].unique())
                        OWN_CAR_AGE_col15 = st.selectbox(label = "Age of client's car",options=without_en[without_en[best_param["COL"]].select_dtypes(["int32","int64","float32","float64"]).columns[11]].unique())
                        LIVINGAREA_MODE_col17 = st.selectbox(label="where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor",options=without_en[without_en[best_param["COL"]].select_dtypes(["int32","int64","float32","float64"]).columns[12]].unique())
                        AMT_GOODS_PRICE_x_col11 = st.number_input(label="Goods price of good that client asked for (if applicable) on the previous application",max_value=1341000.0,min_value=40500.0)
                        AMT_REQ_CREDIT_BUREAU_YEAR_col9 = st.selectbox(label= "Number of enquiries to Credit Bureau about the client one day year (excluding last 3 months before application)",options=without_en[without_en[best_param["COL"]].select_dtypes(["int32","int64","float32","float64"]).columns[8]].unique())

        with colq3:
            if u_name == "Shanth" and pass_ == "Shashi@007":
                with st.container(height=600):
                    #questions for customer need to be filled by field agent
                    HOUR_APPR_PROCESS_START_x_col10 = st.selectbox(label="Approximately at what day hour did the client apply for the previous application",options=without_en[without_en[best_param["COL"]].select_dtypes(["int32","int64","float32","float64"]).columns[9]].unique())
                    REGION_POPULATION_RELATIVE_col6 = st.number_input(label="Population count on client's Location",max_value=0.072508,min_value=0.00029)
                    EXT_SOURCE_2_col1 = st.number_input(label="Select score for External Source Doc - 2",max_value=0.85,min_value=8.173616518884397e-08)
                    EXT_SOURCE_3_col2 = st.number_input(label="Select score for External Source Doc - 3",max_value=0.89,min_value=0.0005)
                    SK_ID_CURR_col3 = st.selectbox(label="Select Your Application ID",options = without_en[without_en[best_param["COL"]].select_dtypes(["int32","int64","float32","float64"]).columns[2]].unique())

        
        button_ = st.button(label="Loan Status")
        if button_:
            col12_Organisation_type = org_type_best2_dic.get(ORGANIZATION_TYPE_col12)
            col14_occ_type = OCCUPATION_TYPE_.get(OCCUPATION_TYPE_col14)
            col16_family_status = NAME_FAMILY_STATUS_best2.get(NAME_FAMILY_STATUS_col16)
            col13WEEKDAY_APPR_PROCESS_START_x = WEEKDAY_APPR_PROCESS_START_x___.get(WEEKDAY_APPR_PROCESS_START_x_col13)


            X = [EXT_SOURCE_2_col1,EXT_SOURCE_3_col2,SK_ID_CURR_col3,AMT_ANNUITY_x_col4,EXT_SOURCE_1_col5,REGION_POPULATION_RELATIVE_col6,AMT_CREDIT_x_col7,AMT_INCOME_TOTAL_col8,AMT_REQ_CREDIT_BUREAU_YEAR_col9,HOUR_APPR_PROCESS_START_x_col10,AMT_GOODS_PRICE_x_col11,col12_Organisation_type,col13WEEKDAY_APPR_PROCESS_START_x,col14_occ_type,OWN_CAR_AGE_col15,col16_family_status,LIVINGAREA_MODE_col17]
            final_d = np.array(X).reshape(1,-1)
            prediction_ = model_final_.predict(final_d)
            if prediction_[0] == 0:
                with st.container(height=100):
                    st.markdown(f"<h3 style='font-size:30px; color:#FF0000;'>Sorry! As of now we can't provide you a loan</h3>", unsafe_allow_html=True)
            else:
                with st.container(height=100):
                    st.markdown(f"<h3 style='font-size:30px; color:#32CD32;'>Hurray! You're eligible for the Loan</h3>", unsafe_allow_html=True)


# with tab2:
#         with st.expander(label="-",expanded=True):
#             colt2,col23,colt21,colt3 = st.columns([1,1,2.5,0.1])
#             with colt2:
#                 st.write("")
#                 st.subheader("Upload Your File")
#                 st.image("https://media4.giphy.com/media/ZgTR3UQ9XAWDvqy9jv/giphy.gif?cid=790b7611fs8o5esmn1cld7btrv4bz8731z5iny1o91aoi0qe&ep=v1_gifs_search&rid=giphy.gif&ct=g")
#             with col23:
#                 vertical_line = """
#                 <div style="
#                     width: 8px;
#                     height: 300px;
#                     background: linear-gradient(192deg, red, orange, yellow, green, blue, indigo, violet, purple, white);
#                     margin: auto;
#                     border-radius: 10px;">
#                 </div>
#                 """

#                 # Display the vertical line in Streamlit
#                 st.markdown(vertical_line, unsafe_allow_html=True)
#             with colt21:
#                 st.write("")
#                 with st.container(height=300):
#                     name_user = st.text_input("Enter Your Name:")
#                     # pdf_uploaded = st.file_uploader("Upload Your Data or Basis of your GEN-AI as folder",accept_multiple_files=True)
#                     pdf_uploaded = st.file_uploader("Upload a zipped folder",accept_multiple_files=True)
#                     st.write("")
#                     st.write("")
#                     st.divider()
#                     save_folder = "/Users/shanthakumark/Desktop/Sharing/Final_project/uploaded files"
#                     if pdf_uploaded:
#                         for uploaded_file in pdf_uploaded:
#                             # Construct the full file path with the file name
#                             file_path = os.path.join(save_folder, uploaded_file.name)
                            
#                             # Save the file to the specified directory
#                             with open(file_path, 'wb') as f:
#                                 f.write(uploaded_file.getbuffer())
                    


#             st.divider()
#             if name_user and pdf_uploaded:
#                 st.progress(100, text="Loaded...")
#             elif name_user or pdf_uploaded:
#                 st.progress(40, text="Processing...")

#         if name_user and pdf_uploaded:
#             with st.container(height=550):
#                 colt2_,colt21_,colt3_ = st.columns([1,1,1])
#                 with colt2_:
#                     st.header(f"Hello {name_user} 🤗.....",divider="rainbow")
                    
                    

#                 with st.container(height=100):
                    
#                     input_query = st.text_input("Enter Your Queries")
#                 if input_query:
#                     start = time.time()
#                     with st.spinner("Processing..."):
#                         answer = retr_answer(input_query)
#                         with st.container(height=100):
#                             st.header(f"{answer}")
#                             end = time.time()
#                             st.write(f"Time taken: {end - start} seconds")
                
#                 with colt21_:
#                     header_placeholder = st.empty()
#                     for j in range(10):
#                         for i in name_user:
#                             time.sleep(0.3)
#                             print(i,end = "")


with tab3:
    st.title("Human Detection App")
    st.write("Upload an image to detect humans. Non-human objects will be ignored.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)

    # Load YOLOv5 model
    model = load_model()

    # Detect humans in the image
    processed_image, human_count = detect_humans(image, model)

    # Display original and processed images side by side
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(height = 800):
            st.header("Uploaded Image",divider = 'rainbow')
            st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        with st.container(height = 800):
            st.header("Processed Image",divider = 'rainbow')
            st.image(processed_image, caption=f"Processed Image - {human_count} human(s) detected", use_column_width=True)


with tab4:
    main()