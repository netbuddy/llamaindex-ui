import streamlit as st
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
import os
from llama_index.llms.openai import OpenAI
import tempfile
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.index_store.redis import RedisIndexStore
from llama_index.vector_stores.milvus import MilvusVectorStore

# 加载环境变量
load_dotenv()

import openai
# 获取API Key和Base URL
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.base_url = os.getenv("OPENAI_BASE_URL")

# 获取Redis配置
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# 设置页面配置
st.set_page_config(
    page_title="LlamaIndex UI",
    page_icon="🦙",
    layout="wide"
)

class LlamaIndexUI:
    def __init__(self):
        # 初始化session state
        if 'current_index' not in st.session_state:
            st.session_state.current_index = None
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        if 'temp_dir' not in st.session_state:
            st.session_state.temp_dir = tempfile.mkdtemp()
        # 初始化工具配置的session state
        if 'milvus_config' not in st.session_state:
            st.session_state.milvus_config = {
                'host': 'localhost',
                'port': '19530'
            }
        if 'redis_config' not in st.session_state:
            st.session_state.redis_config = {
                'host': REDIS_HOST,
                'port': REDIS_PORT,
                'password': '',
                'db': 0,
                'namespace': 'llama_index'
            }

    def main(self):
        # 侧边栏导航
        st.sidebar.title("🦙 LlamaIndex UI")
        page = st.sidebar.radio(
            "导航",
            ["首页", "工具配置", "文档与索引管理", "检索测试"]
        )
        
        # 页面路由
        if page == "首页":
            self.home_page()
        elif page == "工具配置":
            self.tools_config_page()
        elif page == "文档与索引管理":
            self.document_and_index_page()
        elif page == "检索测试":
            self.query_page()

    def home_page(self):
        st.title("欢迎使用 LlamaIndex UI")
        st.write("""
        这是一个基于Streamlit的LlamaIndex可视化操作界面。
        
        主要功能：
        - 文档管理：上传管理您的文档
        - 索引管理：创建和配置索引
        - 检索测试：测试您的索引检索效果
        """)

    def document_and_index_page(self):
        st.title("文档与索引管理")
        
        # 设置知识库索引名称
        namespace = st.text_input("知识库索引名称", value=st.session_state.redis_config['namespace'])
        
        # 文件上传部分
        uploaded_files = st.file_uploader(
            "上传文档",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'docx']
        )
        
        if uploaded_files:
            for file in uploaded_files:
                file_path = os.path.join(st.session_state.temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                if file.name not in st.session_state.uploaded_files:
                    st.session_state.uploaded_files.append(file.name)
        
        # 创建Redis存储上下文
        storage_context = StorageContext.from_defaults(
            docstore=RedisDocumentStore.from_host_and_port(
                host=st.session_state.redis_config['host'],
                port=int(st.session_state.redis_config['port']),
                namespace=namespace
            ),
            index_store=RedisIndexStore.from_host_and_port(
                host=st.session_state.redis_config['host'],
                port=int(st.session_state.redis_config['port']),
                namespace=namespace
            ),
            vector_store=MilvusVectorStore(
                uri="tcp://" + st.session_state.milvus_config['host'] + ":" + st.session_state.milvus_config['port'],
                collection_name=namespace,
                dim=1536,
                overwrite=st.session_state.milvus_config.get('overwrite', True)
            )
        )
        
        # 存储文档到Redis时,将索引名称添加到列表中
        if uploaded_files:
            try:
                # 获取 Redis 连接
                import redis
                redis_client = redis.Redis(
                    host=st.session_state.redis_config['host'],
                    port=int(st.session_state.redis_config['port']),
                    password=st.session_state.redis_config.get('password', ''),
                    db=st.session_state.redis_config.get('db', 0)
                )
                
                # 将索引名称添加到集合中
                redis_client.sadd('llama_index:namespaces', namespace)
                redis_client.close()
                
                # 读取文档
                documents = SimpleDirectoryReader(st.session_state.temp_dir).load_data()
                # 解析文档为节点
                nodes = SentenceSplitter().get_nodes_from_documents(documents)
                # 添加到Redis docstore
                storage_context.docstore.add_documents(nodes)
                st.success("成功将文档存储到Redis中！")
            except Exception as e:
                st.error(f"存储文档时发生错误: {str(e)}")
        
        # 显示已上传的文件
        if st.session_state.uploaded_files:
            st.subheader("已上传的文件")
            for file in st.session_state.uploaded_files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(file)
                with col2:
                    if st.button("删除", key=f"del_{file}"):
                        file_path = os.path.join(st.session_state.temp_dir, file)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        st.session_state.uploaded_files.remove(file)
                        st.experimental_rerun()

        # 索引配置部分
        st.divider()
        with st.form("index_config"):
            st.subheader("索引配置")
            
            # Embedding模型配置
            col1, col2 = st.columns(2)
            with col1:
                embedding_type = st.selectbox(
                    "选择Embedding模型",
                    ["OpenAI Embedding", "BAAI/bge-base-zh-v1.5"]
                )
            with col2:
                if embedding_type == "OpenAI Embedding":
                    # 移除界面中的OpenAI API Key输入
                    st.write("OpenAI API Key 已通过 .env 文件读取")
                    # 使用 .env 中的 API Key
                    embed_model = OpenAIEmbedding()
                else:
                    st.write("使用本地模型，无需API Key")
                    embed_model = HuggingFaceEmbedding(
                        model_name="BAAI/bge-base-zh-v1.5"
                    )
            
            # 解析设置
            st.subheader("解析设置")
            col1, col2 = st.columns(2)
            with col1:
                chunk_size = st.number_input(
                    "分块大小",
                    min_value=100,
                    max_value=2000,
                    value=512
                )
            with col2:
                chunk_overlap = st.number_input(
                    "分块重叠",
                    min_value=0,
                    max_value=500,
                    value=50
                )
            
            submitted = st.form_submit_button("创建索引")
            
            if submitted:
                if not st.session_state.uploaded_files:
                    st.error("请先上传文件")
                    return
                
                try:
                    llm = OpenAI(llm="gpt-3.5-turbo-0125")
                    Settings.llm = llm
                    Settings.chunk_size=chunk_size,
                    Settings.chunk_overlap=chunk_overlap,
                    # Settings.embed_model=embed_model
                    # embed_model = HuggingFaceEmbedding(
                    #     model_name="BAAI/bge-base-zh-v1.5"
                    # )
                    # Settings.embed_model = embed_model

                    # 创建索引并存储到Redis
                    index = VectorStoreIndex(
                        nodes,
                        storage_context=storage_context
                    )
                    st.session_state.current_index = index
                    st.success("索引创建成功并存储到Redis！")
                except Exception as e:
                    st.error(f"创建索引时发生错误: {str(e)}")

    def query_page(self):
        st.title("检索测试")
        
        try:
            # 从 Redis 获取所有知识库索引名称
            import redis
            redis_client = redis.Redis(
                host=st.session_state.redis_config['host'],
                port=int(st.session_state.redis_config['port']),
                password=st.session_state.redis_config.get('password', ''),
                db=st.session_state.redis_config.get('db', 0)
            )
            
            # 从集合中获取所有索引名称
            namespaces = list(redis_client.smembers('llama_index:namespaces'))
            redis_client.close()
            
            # 将字节转换为字符串
            namespaces = [ns.decode() for ns in namespaces]
            
            if not namespaces:
                st.warning("未找到任何知识库索引，请先创建索引")
                return
            
            # 选择知识库索引
            selected_namespace = st.selectbox(
                "选择知识库索引",
                options=namespaces,
                key="selected_namespace"
            )
            
            # 添加加载按钮
            if st.button("加载选中的知识库"):
                with st.spinner("正在加载知识库索引..."):
                    try:
                        # 加载选中的索引
                        storage_context = StorageContext.from_defaults(
                            docstore=RedisDocumentStore.from_host_and_port(
                                host=st.session_state.redis_config['host'],
                                port=int(st.session_state.redis_config['port']),
                                namespace=selected_namespace
                            ),
                            index_store=RedisIndexStore.from_host_and_port(
                                host=st.session_state.redis_config['host'],
                                port=int(st.session_state.redis_config['port']),
                                namespace=selected_namespace
                            ),
                            vector_store=MilvusVectorStore(
                                uri="tcp://" + st.session_state.milvus_config['host'] + ":" + st.session_state.milvus_config['port'],
                                collection_name=selected_namespace,
                                dim=1536,
                                overwrite=False
                            )
                        )
                        
                        # 加载索引
                        index = load_index_from_storage(storage_context)
                        
                        # 将加载的索引存储在 session state 中
                        st.session_state.loaded_index = index
                        st.success(f"知识库 '{selected_namespace}' 加载成功！")
                        
                    except Exception as e:
                        st.error(f"加载知识库索引时发生错误: {str(e)}")
                        st.error(f"错误详情: {str(e.__class__.__name__)}")
                        return
            
            # 只有在索引加载成功后才显示查询部分
            if 'loaded_index' in st.session_state:
                # 查询输入
                query = st.text_input("输入查询内容")
                
                if st.button("搜索"):
                    if not query:
                        st.warning("请输入查询内容")
                        return
                    
                    try:
                        # 创建查询引擎
                        query_engine = st.session_state.loaded_index.as_query_engine()
                        
                        # 执行查询
                        response = query_engine.query(query)
                        
                        # 显示结果
                        st.subheader("查询结果")
                        st.write(response.response)
                        
                        # 显示源文档
                        st.subheader("相关文档片段")
                        for node in response.source_nodes:
                            with st.expander(f"文档片段 (相似度: {node.score:.2f})"):
                                st.write(node.node.text)
                                
                    except Exception as e:
                        st.error(f"查询时发生错误: {str(e)}")
                        st.error(f"错误详情: {str(e.__class__.__name__)}")
                    
        except Exception as e:
            st.error(f"获取知识库索引列表时发生错误: {str(e)}")
            st.error(f"错误详情: {str(e.__class__.__name__)}")

    def tools_config_page(self):
        st.title("工具配置")
        
        # Milvus配置
        st.header("Milvus 配置")
        milvus_col1, milvus_col2 = st.columns(2)
        with milvus_col1:
            milvus_host = st.text_input(
                "Milvus主机",
                value=st.session_state.milvus_config['host'],
                key="milvus_host"
            )
        with milvus_col2:
            milvus_port = st.text_input(
                "Milvus端口",
                value=st.session_state.milvus_config['port'],
                key="milvus_port"
            )
        
        # Redis配置
        st.header("Redis 配置")
        redis_col1, redis_col2 = st.columns(2)
        with redis_col1:
            redis_host = st.text_input(
                "Redis主机",
                value=st.session_state.redis_config['host'],
                key="redis_host"
            )
            redis_password = st.text_input(
                "Redis密码",
                value=st.session_state.redis_config['password'],
                type="password",
                key="redis_password"
            )
            redis_namespace = st.text_input(
                "Redis命名空间",
                value=st.session_state.redis_config['namespace'],
                key="redis_namespace"
            )
        with redis_col2:
            redis_port = st.text_input(
                "Redis端口",
                value=st.session_state.redis_config['port'],
                key="redis_port"
            )
            redis_db = st.number_input(
                "Redis数据库",
                value=st.session_state.redis_config['db'],
                min_value=0,
                max_value=15,
                key="redis_db"
            )
        
        # 保存配置按钮
        if st.button("保存配置"):
            # 更新Milvus配置
            st.session_state.milvus_config = {
                'host': milvus_host,
                'port': milvus_port
            }
            
            # 更新Redis配置
            st.session_state.redis_config = {
                'host': redis_host,
                'port': redis_port,
                'password': redis_password,
                'db': redis_db,
                'namespace': redis_namespace
            }
            
            st.success("配置已保存！")
            
        # 测试连接按钮
        if st.button("测试Redis连接"):
            try:
                # 测试Redis连接
                docstore = RedisDocumentStore.from_host_and_port(
                    host=redis_host,
                    port=int(redis_port),
                    password=redis_password or None,
                    namespace=redis_namespace
                )
                index_store = RedisIndexStore.from_host_and_port(
                    host=redis_host,
                    port=int(redis_port),
                    password=redis_password or None,
                    namespace=redis_namespace
                )
                st.success("Redis连接测试成功！")
            except Exception as e:
                st.error(f"Redis连接测试失败: {str(e)}")

if __name__ == "__main__":
    app = LlamaIndexUI()
    app.main() 