import os
# Hugging Face の Xet/CAS 経由を無効化
os.environ["HF_HUB_DISABLE_XET"] = "1"

# 余計な高速転送も切る（環境によって不安定要因）
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# タイムアウトを延ばす（ネットワークが遅い環境向け）
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
import streamlit as st
import chromadb
import base64
import mimetypes
from pathlib import Path
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


def main():
    st.set_page_config(page_title="ベxxxxト2025年IRチャットボット", page_icon=":robot_face:")
    st.header("ベxxxxト2025年IRチャットボット")
    user_input = st.chat_input("ベxxxxト2025年IRについて質問してみてください: ")

    # Chromaクライアントのセットアップ
    chroma_collection = set_chroma_client()

    if user_input:
        st.chat_message("user").write(user_input)

        # RAGの呼び出し
        rag_response = text_rag(chroma_collection, user_input)

        # 画像RAGの呼び出し
        image_responses = image_rag(chroma_collection, user_input)
            # デバグ用　st.write(image_responses)

        # 画像の説明文を取得してRAGレスポンスに追加
        image_captions = ""
        for image in image_responses:
            r_image_path = fr"{image}"
            caption = caption_image(r_image_path)
            image_captions += f"\n画像説明: {caption}" 
            # デバグ用　st.write(image_captions)

        # LLMの呼び出し
        llm_response = response_generator(user_input, rag_response, image_captions)

        # チャットに回答を表示
        st.chat_message("assistant").write(llm_response)

        # 画像RAGの回答も表示
        for image in image_responses:
            st.chat_message("assistant").image(image)


# RAG検索先のChroma DBクライアントをセットアップ
def set_chroma_client():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    data_loader = ImageLoader() # OpenCLIPでは直接画像をChroma DBに保存しない。そのため、URI指定で生画像を持ってきてくれるImage Loaderを使用する。
    embedding_function = OpenCLIPEmbeddingFunction()

    collection = chroma_client.get_collection(
        name='multimodal_collection', 
        data_loader=data_loader,
        embedding_function=embedding_function
    )
    return collection


def text_rag(collection, user_input) -> str:
    results = collection.query(
        query_texts=[user_input], 
        n_results=5, # 返す件数
        include=["documents"] # 返す内容指定. dataを入れるとData loaderが自動的に呼び出され、（画像などの）データも返す。
        )
    
    text_retrieve1 = results["documents"][0][0]
    text_retrieve2 = results["documents"][0][1] 
    text_retrieve3 = results["documents"][0][2]
    text_retrieve4 = results["documents"][0][3]
    text_retrieve5 = results["documents"][0][4]

    response = f"1. {text_retrieve1}\n2. {text_retrieve2}\n3. {text_retrieve3}\n4. {text_retrieve4}\n5. {text_retrieve5}"
    return response


def image_rag(collection, user_input) -> list:
    results = collection.query(
        query_texts=[user_input], 
        n_results=3, # 返す件数
        include=["documents","data"], # 返す内容指定. dataを入れるとData loaderが自動的に呼び出され、（画像などの）データも返す。
        where={"modality": "image"} # 画像データのみ絞り込み
    )
    docs = results["documents"]

    image_pass0 = "./" + docs[0][0]
    image_pass1 = "./" + docs[0][1]
    image_pass2 = "./" + docs[0][2]
    image_paths = [image_pass0, image_pass1, image_pass2]
    return image_paths


# TODO:キャプション生成の関数を追加
def _image_path_to_data_url(image_path: str | Path) -> str:
    """画像ファイルを base64 の data URL に変換（MIME 自動判定）"""
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None:
        # 判定不能時の保険（多くはここに来ない）
        mime_type = "application/octet-stream"

    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def caption_image(image_path: str) -> str:
    """
    インプット: 画像パス（例: r".\\.image\\xxxxx.png"）
    アウトプット: 画像のキャプション（日本語の短文）
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set in environment variables.")

    data_url = _image_path_to_data_url(image_path)

    llm = ChatOpenAI(model_name="gpt-5.2", 
                     temperature=0, 
                     openai_api_key=os.environ["OPENAI_API_KEY"]
                    )
    # 画像 + テキストを同一メッセージに含める（Vision入力）
    messages = [
        {
            "role": "system",
            "content": "あなたは株式会社ベイカレントのIR担当者です。いまから2025年度IR資料のpng画像を見せるので、その内容を簡潔で自然な日本語の400文字で説明してください。",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "この画像の内容をキャプションとして400文字で書いてください。出力はその400文字のみです。"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]

    result = llm.invoke(messages)
    return result.content.strip()


def response_generator(user_input: str, rag_response: str, image_captions: str) -> str:
    prompt = prompt_bay_ir_2025()
    llm = set_llm()
    chain = prompt | llm

    response = chain.invoke(
        {
            "user_input": user_input,
            "rag_input": rag_response, 
            "image_captions": image_captions
        }
    )
    
    return response.content


def prompt_bay_ir_2025():
    prompt = PromptTemplate(
        input_variables=["user_input", "rag_input", "image_captions"],
        template="""
        あなたはベイカレント株式会社のIR担当者です。

        以下はベイカレントの2025年決済に関する情報です。
        {rag_input}

        以下は画像の説明文です。
        {image_captions}
        
        上記の情報をもとに、以下の質問に答えてください。
        質問: {user_input}
        """)
    return prompt


def set_llm():
    llm = ChatOpenAI(model_name="gpt-5.2", 
                     temperature=0, 
                     openai_api_key=os.environ["OPENAI_API_KEY"]
                    )
    return llm

if __name__ == "__main__":
    load_dotenv()
    main()