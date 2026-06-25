import numpy as np 
import streamlit as st

from sklearn.metrics.pairwise import cosine_similarity 
#thư viện tính cosine similarity của vector
from sentence_transformers import SentenceTransformer
#sentence_transformers thư viẹn dùng chuyển văn bản thành embedding vector

st.title("RAG Search Demo")

@st.cache_resource
#Khởi tạo mô hình embedding chuyên dụng cho Tiếng Việt
def load_model():
    return SentenceTransformer('keepitreal/vietnamese-sbert')

model = load_model()
###  INDEXING
#Cơ sở dữ liệu văn bản 
docs = [
    "Khoá học AIO2026 gồm có 10 module chuyên sâu về Trí tuệ nhân tạo.",
    "Tài liệu hướng dẫn thủ tục đăng kí nhập học và đóng học phí trực tuyến.",
    "Hướng dẫn cấu trúc dữ luệu và giải thuật với Python để xử lý mảng."
]

#Mô hình tự chuyển động văn bản thành ma trận vector n-chiều
vector_db = model.encode(docs)

#Câu hỏi người dùng
query = st.text_input("Khóa học AIO2026/cấu trúc dữ liệu và/nhập học va đóng học phí")
if st.button("Tìm kiếm"):
    query_vector = model.encode(query)

    #tính toán điểm Cosine Similarity trên toàn ma trận vector_db
    scores = cosine_similarity([query_vector], vector_db)[0]

###Hoặc có thể dùng công thức đối với người chưa biết về cosine similarity như sau:
#Công thức : (A dot B) / (||A|| * ||B||)
#dot_products = np.dot(vector_db,query_vector).    #tích vô hướng
#norm_db = np.linalg.norm(vector_db, axis = 1).    #độ dài vector database(văn bản cơ sở)
#norm_query  = np.linalg.norm(query_vector).       #Độ dài vector truy vấn(văn bản người dùng nhập)
#scores = dot_products / (norm_db * norm_query)     #Tính cosine Similarity

#Trích xuất đoạn văn bản có điểm(scores) cao nhất
    best_match_idx = np.argmax(scores)# np.argmax : tìm index của phần tử lớn nhất mảng
    retrieved_context = docs[best_match_idx] 
    st.subheader("Kết quả Retrieval")
    st.write(
        "Điểm tương đồng" ,
        [round(float(s),4) for s in scores]    
    )
    st.success(retrieved_context)

    ###  GENERATION
    #Kết hợp dữ liệu thành cấu trúc Prompt quy định
    prompt_template = f"""Dựa vào ngữ cảnh được cung cấp dưới đây để trả lời câu hỏi.\nTuyệt đối không sử dụng thông tin bên ngoài vì lượng dữ liệu hạn chế.

    Câu hỏi: {query}
    Trả lời: {retrieved_context}
    """

    st.code(prompt_template)

