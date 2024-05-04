import csv
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import os
import streamlit as st
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import GPTVectorStoreIndex, download_loader
from llama_index.readers.file import CSVReader
from pathlib import Path
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import PromptTemplate
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import BaseSynthesizer

from llama_index.core import VectorStoreIndex

os.environ['OPENAI_API_KEY'] = ''

def remove_last_n_columns(csv_file, n):
    # Đọc dữ liệu từ tập tin CSV và loại bỏ n cột cuối cùng từ mỗi dòng
    with open(csv_file, 'r', encoding="utf8") as file:
        csv_reader = csv.reader(file)
        data = [[cell.lower().replace('- ','-').replace(' -','-') for cell in row[:-n]] for row in csv_reader]  # Chuyển đổi thành chữ thường và xóa bỏ cột cuối cùng

    # Ghi dữ liệu đã chỉnh sửa vào tập tin CSV mới
    with open('new_csv_file.csv', 'w', newline='', encoding="utf8") as new_file:
        csv_writer = csv.writer(new_file)
        csv_writer.writerows(data)


# Số lượng cột cuối cùng cần xóa bỏ
n_columns_to_remove = 9
csv_file_path = '113_final_oke.csv'

# Gọi hàm để xóa bỏ n cột cuối cùng
remove_last_n_columns(csv_file_path, n_columns_to_remove)
import csv
import sqlite3

# Tên của tập tin CSV
csv_file = 'new_csv_file.csv'

# # Kết nối đến cơ sở dữ liệu SQLite hoặc tạo một cơ sở dữ liệu mới
# conn = sqlite3.connect('database.db')
# cursor = conn.cursor()
#
# # Tạo bảng trong cơ sở dữ liệu
# cursor.execute('''CREATE TABLE IF NOT EXISTS du_lieu (
#                     STT INTEGER,
#                     LINK TEXT,
#                     PRODUCT_INFO_ID TEXT,
#                     GROUP_PRODUCT_NAME TEXT,
#                     PRODUCT_CODE TEXT,
#                     PRODUCT_NAME TEXT,
#                     SHORT_DESCRIPTION TEXT,
#                     PRODUCT_INFO TEXT,
#                     SPECIFICATION_BACKUP TEXT,
#                     NON_VAT_PRICE_1 INTEGER,
#                     VAT_PRICE_1 INTEGER,
#                     COMMISSION_1 INTEGER
#                     )''')
#
# # Đọc dữ liệu từ tập tin CSV và chèn vào bảng
# with open(csv_file, 'r', encoding= "utf8") as file:
#     csv_reader = csv.reader(file)
#     next(csv_reader)  # Bỏ qua dòng tiêu đề nếu có
#     for row in csv_reader:
#         cursor.execute('''INSERT INTO du_lieu (STT, LINK, PRODUCT_INFO_ID, GROUP_PRODUCT_NAME, PRODUCT_CODE,  PRODUCT_NAME, SHORT_DESCRIPTION, PRODUCT_INFO,  SPECIFICATION_BACKUP, NON_VAT_PRICE_1, VAT_PRICE_1, COMMISSION_1)
#                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', row)
#
#
# # Lưu thay đổi và đóng kết nối
# conn.commit()
# conn.close()


loader = CSVReader()

# load gdocs and index them
file_path = Path("113_final_oke.csv")
# documents = loader.load_data(file_path)
# index = GPTVectorStoreIndex(documents)
#
import csv
import re
def norm_text(text):
    text = text.lower().replace('- ', '-').replace(' -', '-').replace('  ',' ')
    normalized_text = re.sub(r'\s+', ' ', text)
    # print(normalized_text)
    return normalized_text.strip()

# Đọc file CSV
def csv2txt(csv_link):
    data_text = ''
    with open(csv_link, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Lấy thông tin từ mỗi hàng của file CSV
            name = row['PRODUCT_NAME']  # Thay 'Tên Sản Phẩm' bằng tên cột chứa tên sản phẩm trong file CSV của bạn
            id = row['PRODUCT_INFO_ID']  # Thay 'ID' bằng tên cột chứa ID sản phẩm trong file CSV của bạn
            code = row['PRODUCT_CODE']  # Thay 'Code' bằng tên cột chứa mã code sản phẩm trong file CSV của bạn
            group = row['GROUP_PRODUCT_NAME']  # Thay 'Nhóm' bằng tên cột chứa nhóm sản phẩm trong file CSV của bạn
            s_des = row['SHORT_DESCRIPTION']
            p_inf = row['PRODUCT_INFO']
            link = row['LINK_SP']
            nv1 = row['NON_VAT_PRICE_1']
            v1 = row['VAT_PRICE_1']
            comm1 = row['COMMISSION_1']
            nv2 = row['NON_VAT_PRICE_2']
            v2 = row['VAT_PRICE_2']
            comm2 = row['COMMISSION_2']
            nv3 = row['NON_VAT_PRICE_3']
            v3 = row['VAT_PRICE_3']
            comm3 = row['COMMISSION_3']
            thong_so = row['SPECIFICATION_BACKUP']
            # In ra văn bản theo định dạng mong muốn
            s = f"Sản phẩm \"{name}\" có ID {id} và Code {code} thuộc nhóm {group} là {s_des}, thông tin chi tiết về sản phẩm \"{name}\" : {p_inf}. Thông số : {thong_so}. Link của sản phẩm \"{name}\" là {link}. Về giá của sản phẩm \"{name}\": Nếu tổng giá trị đơn hàng trên 30 triệu thì giá sản phẩm \"{name}\" không bao gồm VAT là {nv1}, giá sản phẩm \"{name}\" bao gồm VAT là {v1} và tiền hoa hồng sản phẩm \"{name}\" là {comm1}, Nếu tổng giá trị đơn hàng từ 15 đến 30 triệu thì giá sản phẩm \"{name}\" không bao gồm VAT là {nv2}, giá sản phẩm \"{name}\" bao gồm VAT là {v2} và tiền hoa hồng sản phẩm \"{name}\" là {comm2}, Nếu tổng giá trị đơn hàng dưới 15 triệu thì giá sản phẩm \"{name}\" không bao gồm VAT là {nv3}, giá sản phẩm \"{name}\" bao gồm VAT là {v3} và tiền hoa hồng sản phẩm \"{name}\" là {comm3}."
            s = s.replace('\n', ' ')
            s = s.replace('  ', ',')
            s = s.replace('..', ',')
            data_text = data_text + s + '\n'
            # print(s)
    return data_text


data_text = csv2txt('113_final_oke.csv')
data_text = norm_text(data_text)
from llama_index.core import Document

doc = Document(text=data_text)
documents = []
documents.append(doc)


Settings.llm = OpenAI(model="gpt-3.5-turbo",temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=40)
Settings.num_output = 512
Settings.context_window = 15000
splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

nodes = splitter.get_nodes_from_documents(documents)
# index = VectorStoreIndex(nodes)
embed_model = OpenAIEmbedding(model="text-embedding-3-small", dimensions=1024)
llm = OpenAI(model="gpt-3.5-turbo", temperature=0)


# index.storage_context.persist(persist_dir="db")
storage_context = StorageContext.from_defaults(persist_dir="db")

# load index
index = load_index_from_storage(storage_context)
few_shot_examples = """
"Query": "có bao nhiêu loại nồi chiên"
"Answer": SELECT COUNT(*) \nFROM du_lieu \nWHERE PRODUCT_NAME LIKE \'%nồi chiên%\';

"Query": "so sánh Bếp Từ Đôi bluestone icb-6948 với bếp hỗn hợp quang từ bluestone icb-6911"
"Answer": SELECT * \nFROM du_lieu \nWHERE PRODUCT_NAME LIKE '%icb-6948%' OR PRODUCT_NAME LIKE '%icb-6911%';

"Query": "điều hòa nào rẻ nhất"
"Answer": SELECT PRODUCT_NAME  VAT_PRICE_1\nFROM du_lieu\nWHERE GROUP_PRODUCT_NAME LIKE '%điều hòa%'\nORDER BY VAT_PRICE_1 ASC\nLIMIT 1;

"Query": "bên bạn có những mặt hàng nào"
"Answer": SELECT DISTINCT GROUP_PRODUCT_NAME \nFROM du_lieu;

"Query": "dung tích của bếp từ đôi bluestone icb-6948"
"Answer": SELECT * \nFROM du_lieu \nWHERE PRODUCT_NAME LIKE \'%icb-6948%\'\nLIMIT 1;


"Query": "đèn aiosmart ad300 có thể thắp sáng liên tục trong bao lâu"
"Answer": SELECT * \nFROM du_lieu \nWHERE PRODUCT_NAME LIKE \'%aiosmart ad300%\'\nLIMIT 1;

"Query": "Bên bạn có bán các sản phẩm nào"
"Answer": SELECT DISTINCT GROUP_PRODUCT_NAME \nFROM du_lieu;

"Query": "có hình ảnh nồi kl-619 không"
"Answer": SELECT * \nFROM du_lieu \nWHERE PRODUCT_NAME LIKE \'%kl-619%\'\nLIMIT 1;


"Query": "bếp từ đôi bluestone icb-6845 đắt quá, có cái nào rẻ hơn không"
"Answer": SELECT PRODUCT_NAME  VAT_PRICE_1 \nFROM du_lieu \nWHERE GROUP_PRODUCT_NAME LIKE '%bếp từ%'\nORDER BY VAT_PRICE_1 ASC\nLIMIT 3;


"Query": "có loại nào rẻ hơn nồi chiên không dầu kalite Q6 5 5 lít không"
"Answer": SELECT PRODUCT_NAME  VAT_PRICE_1 \nFROM du_lieu \nWHERE GROUP_PRODUCT_NAME LIKE '%nồi chiên không dầu%'\nORDER BY VAT_PRICE_1 ASC\nLIMIT 3;

"Query": "tôi có 10 triệu và tôi muốn mua nồi cơm và bình nước nóng"
"Answer": SELECT PRODUCT_NAME, NON_VAT_PRICE_1, VAT_PRICE_1\nFROM du_lieu \nWHERE VAT_PRICE_1 <= 10000000 \nAND PRODUCT_NAME LIKE '%nồi cơm%'\nUNION\nSELECT PRODUCT_NAME, NON_VAT_PRICE_1, VAT_PRICE_1 \nFROM du_lieu \nWHERE VAT_PRICE_1 <= 10000000 \nAND PRODUCT_NAME LIKE '%bình nước nóng%';
"""
# write prompt template with functions


qa_prompt_tmpl_str = """
Bạn là một chuyên gia SQLite. Đưa ra một câu hỏi đầu vào, hãy tạo một truy vấn SQLite đúng về mặt cú pháp để chạy. Sử dụng 'LIKE' thay vì '='.
Nếu sử dụng "SELECT *" không trả về nhiều hơn 3 hàng, các trường hợp còn lại trả về dưới 10 hàng.
Hỏi về thông tin một sản phẩm thì nên trả ra tất cả các cột của sản phẩm đó.
Dưới đây là thông tin các cột của bảng du_lieu trong cơ sở dữ liệu:
LINK_SP,PRODUCT_INFO_ID,GROUP_PRODUCT_NAME,PRODUCT_CODE,PRODUCT_NAME,SHORT_DESCRIPTION,PRODUCT_INFO,SPECIFICATION_BACKUP,NON_VAT_PRICE_1,VAT_PRICE_1,COMMISSION_1
Chú ý tham khảo thêm thông tin dưới đây để xác định đúng trường PRODUCT_NAME và GROUP_PRODUCT_NAME nếu người dùng nhập sai:
---------------------
{context_str}
---------------------
Một số ví dụ được đưa ra dưới đây:

{few_shot_examples}

Query: {query_str}
Answer:
"""
additional_args = {"context_str": "{context_str}", "query_str": "{query_str}", "few_shot_examples": few_shot_examples}
qa_prompt_tmpl = PromptTemplate(
    qa_prompt_tmpl_str,
    # format =additional_args,
)
qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl.format(**additional_args))

# build index

# configure retriever
retriever = index.as_retriever()

# configure response synthesizer
response_synthesizer = get_response_synthesizer()

# assemble query engine
# query_engine = RetrieverQueryEngine(
#     retriever=retriever,
#     response_synthesizer=response_synthesizer,
#     # node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
# )
query_engine_sql = RetrieverQueryEngine.from_args(
    retriever, response_mode='compact'
)
query_engine_sql.update_prompts(
    {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
)
prompts_dict = query_engine_sql.get_prompts()


# display_prompt_dict(prompts_dict)
# query
# response = query_engine_sql.query("nùi cơm nào rẻ nhất")
# print(response)
def run_sql(response):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    # Thực hiện truy vấn
    cursor.execute(response.response.lower())
    # Lấy kết quả truy vấn
    rows = cursor.fetchall()
    # Đóng kết nối
    conn.close()
    # print(str(rows))
    return str(rows)


# run_sql(response)


qa_prompt = PromptTemplate(
    "Đưa ra câu hỏi của người dùng sau với tư cách là trợ lý bán hàng, sử dụng thông tin từ truy vấn SQL để trả lời câu hỏi của người dùng. Nếu không có thông tin hãy in ra dòng chữ 'Không có thông tin.'. Lưu ý khi trả lời về giá thì nói rõ giá đấy là có VAT hay không VAT\n"
    "---------------------\n"
    "SQL Query: {sql_query}"
    "SQL Result: {sql_result}"
    "---------------------\n"
    "Sử dụng thông tin từ truy vấn SQL để trả lời câu hỏi của người dùng\n"
    "Lưu ý: nếu người dùng chào hỏi giao tiếp bình thường chỉ cần đáp lại mà không cần dựa vào kết quả sql trên"
    "Query: {query_str}\n"
    "Answer: "
)


class RAGStringQueryEngine(CustomQueryEngine):
    """SQL Query Engine."""

    # retriever: BaseRetriever
    # response_synthesizer: BaseSynthesizer
    llm: llm
    qa_prompt: PromptTemplate


    def custom_query(self, query_str: str, sql_query: str, sql_result: str):
        # nodes = self.retriever.retrieve(query_str)

        # context_str = "\n\n".join([n.node.get_content() for n in nodes])
        response = self.llm.complete(
            qa_prompt.format(query_str=query_str, sql_query=sql_query, sql_result=sql_result)
        )

        return response
from llama_index.core.tools import FunctionTool
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool

def retriever_syns(query: str) -> str:
    """đưa chi tiết câu hỏi khách hàng vào đây"""
    # response = query_engine_sql.query(query)
    # print(response.response)
    # result = run_sql(response)
    # print(result)
    context_str = '{context_str}'
    query_str = '{query_str}'
    final_prompt_tmpl_str = f"""
    Đưa ra câu hỏi của người dùng sau với tư cách là trợ lý bán hàng. Nếu không có thông tin hãy trả lời là không có thông tin. Lưu ý khi trả lời về giá thì nói rõ giá đấy là có VAT hay không VAT
    Context: {context_str}
    Query: {query_str}
    Answer:
    """
    # additional_args = {"context_str": "{context_str}", "query_str": "{query_str}", "few_shot_examples": few_shot_examples}
    final_prompt_tmpl = PromptTemplate(
        final_prompt_tmpl_str,
        # format =additional_args,
    )
    retriever = index.as_retriever()

    # configure response synthesizer
    response_synthesizer = get_response_synthesizer()
    query_engine_retriever = RetrieverQueryEngine.from_args(
    retriever, response_mode='compact'
    )
    query_engine_retriever.update_prompts(
    {"response_synthesizer:text_qa_template": final_prompt_tmpl}
    )
    # prompts_dict = query_engine_retriever.get_prompts()
    # display_prompt_dict(prompts_dict)
    # prompts_dict = query_engine.get_prompts()
    # display_prompt_dict(prompts_dict)
    # query
    ans = query_engine_retriever.query(query)

    return ans.response


retriever_tool = FunctionTool.from_defaults(fn=retriever_syns, return_direct=True)
memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
def product_tool(query: str) -> str:
    """Trả lời bất cứ câu hỏi nào của khách hàng, lấy toàn bộ câu hỏi của người dùng truyền vào đây, không được bỏ sót một từ nào"""
    print(query)
    try:
        response = query_engine_sql.query(query)
        print(response.response)
        sql_query = response.response
        result = run_sql(response)
        # print(result)
        print(len(result))
        if len(result) > 10000:
            result = result[:10000]
        query_engine = RAGStringQueryEngine(
        # retriever=retriever,
        # response_synthesizer=synthesizer,
        llm=llm,
        qa_prompt=qa_prompt,
        # sql_query=response.response,
        # sql_result=result
        )
        ans = query_engine.custom_query(query, sql_query=sql_query, sql_result=result)
        # prompts_dict = query_engine_retriever.get_prompts()
        # display_prompt_dict(prompts_dict)
        # prompts_dict = query_engine.get_prompts()
        # display_prompt_dict(prompts_dict)
        # query
        # ans = query_engine_retriever.query(query)
        # query_engine_retriever.query()
        answer = ans.text
        print('sql',answer)
    except:
        answer = 'Không có thông tin'
    if 'hông có thông tin' in answer:
        answer = retriever_syns(query)
    return answer
        # return 'Không có thông tin1'


product_tool = FunctionTool.from_defaults(fn=product_tool,return_direct=True)

def Chat_sql_and_retriever(query: str) -> str:
    """lấy toàn bộ câu hỏi của người dùng truyền vào đây, không được bỏ sót một từ nào"""
    response = query_engine_sql.query(query)
    # print(response.response)
    result = run_sql(response)
    # print(result)
    context_str = '{context_str}'
    query_str = '{query_str}'
    final_prompt_tmpl_str = f"""
    Đưa ra câu hỏi của người dùng sau với tư cách là trợ lý bán hàng, kết hợp giữa thông tin từ truy vấn SQL và thông tin ngữ cảnh để trả lời câu hỏi của người dùng. Nếu không có thông tin từ SQL, hãy lấy thông tin từ Context. Nếu không có thông tin từ cả hai hãy trả lời là không có thông tin
    SQL Query: {response.response}
    SQL Result: {result}
    Context: {context_str}
    Query: {query_str}
    Answer:
    """
    # additional_args = {"context_str": "{context_str}", "query_str": "{query_str}", "few_shot_examples": few_shot_examples}
    final_prompt_tmpl = PromptTemplate(
        final_prompt_tmpl_str,
        # format =additional_args,
    )
    retriever = index.as_retriever()

    # configure response synthesizer
    response_synthesizer = get_response_synthesizer()
    query_engine_retriever = RetrieverQueryEngine.from_args(
        retriever, response_mode='compact'
    )
    query_engine_retriever.update_prompts(
        {"response_synthesizer:text_qa_template": final_prompt_tmpl}
    )
    # prompts_dict = query_engine_retriever.get_prompts()
    # display_prompt_dict(prompts_dict)
    # prompts_dict = query_engine.get_prompts()
    # display_prompt_dict(prompts_dict)
    # query
    ans = query_engine_retriever.query(query)

    return ans.response


multiply_tool = FunctionTool.from_defaults(fn=Chat_sql_and_retriever)

loaded_chat_store = SimpleChatStore()

loaded_chat_store = SimpleChatStore.from_persist_path(
    persist_path="chat_store.json"
)

# print('start')
chat_memory = ChatMemoryBuffer.from_defaults(
    token_limit=2000,
    chat_store=loaded_chat_store,
    chat_store_key="user1",
)

# print('aaa')
agent = OpenAIAgent.from_tools(
    [product_tool],
    llm=llm,
    memory=chat_memory,
    verbose=True,
)
original_title = '<h1 style="font-family: serif; color:white; font-size: 20px;">Example</h1>' + \
                 '<h2 style="font-family: serif; color:white; font-size: 15px;">- Nồi chiên không dầu KALITE Q6 5,5 lít với Nồi chiên không dầu Kalite KL-1500 15 lít cái nào xịn hơn' + \
                 '<h2 style="font-family: serif; color:white; font-size: 15px;">- Máy làm sữa hạt Kalite KL-950 có dụng cụ đi kèm không</h2>' + \
                 '<h2 style="font-family: serif; color:white; font-size: 15px;">- Ghế massage nào rẻ nhất</h2>' + \
                 '<h2 style="font-family: serif; color:white; font-size: 15px;">- Có loại nào khoảng từ 1-2 triệu không</h2>'
st.markdown(original_title, unsafe_allow_html=True)
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://img.lovepik.com/background/20211021/medium/lovepik-dark-background-image_400109825.jpg");
    background-size: 100vw 100vh; 
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .reportview-container .main .block-container div[data-baseweb="toast"] {
        background-color: red;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title('Chatbot bán hàng')
with st.form('my_form'):
    text = st.text_area('Input')
    # text = norm_text(text)
    submitted = st.form_submit_button('Run')
    if submitted:
        output = agent.chat(norm_text(text)).response
        loaded_chat_store.persist(persist_path="chat_store.json")
        st.info(output)
        # chat_store.persist(persist_path="chat_store.json")
# app = FastAPI()
# class TextRequest(BaseModel):
#     text: str

# @app.post("/chat/")
# def chat(request: TextRequest):
#     response = agent.chat(request.text)
#     return response.response
# while True:
#     text_input = input("User: ")
#     if text_input == "exit":
#         break
#     response = agent.chat(text_input)
#     print(f"Agent: {response}")
# if __name__ == "__main__":
#     uvicorn.run(app, port=5006, host='0.0.0.0')
# # http://0.0.0.0:5006/chat
