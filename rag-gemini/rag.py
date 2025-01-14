import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
from google import genai
from google.genai import types
import textwrap
from dataclasses import dataclass
from PIL import Image
import time
from ratelimit import limits, sleep_and_retry
import fitz
import io
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

@dataclass
class Config:
    """Configuration class for the application"""
    MODEL_NAME: str = "gemini-2.0-flash-exp"  # Updated to match Google's example
    TEXT_EMBEDDING_MODEL_ID: str = "text-embedding-004"  # Correct embedding model name
    DPI: int = 300  # Resolution for PDF to image conversion

class PDFProcessor:
    """Handles PDF processing using PyMuPDF and Gemini's vision capabilities"""
    
    @staticmethod
    def pdf_to_images(pdf_path: str, dpi: int = Config.DPI) -> List[Image.Image]:
        """Convert PDF pages to PIL Images"""
        images = []
        pdf_document = fitz.open(pdf_path)
        
        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            
            # Convert PyMuPDF pixmap to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
            
        pdf_document.close()
        return images
    

class GeminiClient:
    """Lida com as interações com a API do Gemini"""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("A chave da API é obrigatória")

        # Inicializa o cliente exatamente como no exemplo do Google
        self.client = genai.Client(api_key=api_key)

    def make_prompt(self, element: str) -> str:
        """Criar prompt para resumo"""
        return f"""Você é um agente encarregado de resumir tabelas e textos de pesquisa de artigos científicos para recuperação.
                  Esses resumos serão incorporados e usados para recuperar o texto ou elementos da tabela brutos.
                  Forneça um resumo conciso das tabelas ou texto que seja bem otimizado para recuperação.
                  Tabela ou texto: {element}"""

    def analyze_page(self, image: Image.Image) -> str:
        """Analisar uma página de PDF usando as capacidades visuais do Gemini"""
        prompt = """Você é um assistente encarregado de resumir imagens para recuperação.
                   Esses resumos serão incorporados e usados para recuperar a imagem bruta.
                   Forneça um resumo conciso da imagem que seja bem otimizado para recuperação.
                   Se for uma tabela, extraia todos os elementos da tabela.
                   Se for um gráfico, explique os achados no gráfico.
                   Inclua detalhes sobre cor, proporção e forma, se necessário para descrever a imagem.
                   Extraia todo o conteúdo de texto da página com precisão.
                   Não inclua nenhum número que não seja mencionado na imagem."""

        try:
            response = self.client.models.generate_content(
                model=Config.MODEL_NAME,
                contents=[prompt, image]
            )
            return response.text if response.text else ""
        except Exception as e:
            print(f"Erro ao analisar a página: {e}")
            return ""
        
    @sleep_and_retry
    @limits(calls=30, period=60)
    def create_embeddings(self, data: str):
        """Create embeddings with rate limiting - exactly as in Google's example"""
        time.sleep(1)
        return self.client.models.embed_content(
            model=Config.TEXT_EMBEDDING_MODEL_ID,
            contents=data,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )

    def find_best_passage(self, query: str, dataframe: pd.DataFrame) -> dict:
        """Find the most relevant passage for a query"""
        try:
            query_embedding = self.client.models.embed_content(
                model=Config.TEXT_EMBEDDING_MODEL_ID,
                contents=query,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
            
            dot_products = np.dot(np.stack(dataframe['Embeddings']), 
                                query_embedding.embeddings[0].values)
            idx = np.argmax(dot_products)
            content = dataframe.iloc[idx]['Original Content']
            return {
                'page': content['page_number'],
                'content': content['content']
            }
        except Exception as e:
            print(f"Error finding best passage: {e}")
            return {'page': 0, 'content': ''}
        
    def make_answer_prompt(self, query: str, passage: dict) -> str:
        """Criar prompt para responder perguntas"""
        escaped = passage['content'].replace("'", "").replace('"', "").replace("\n", " ")
        return textwrap.dedent("""Você é um bot útil e informativo que responde perguntas usando o texto do trecho de referência incluído abaixo.
                                Você está respondendo perguntas sobre um artigo de pesquisa.
                                Certifique-se de responder em uma frase completa, sendo abrangente, incluindo todas as informações de fundo relevantes.
                                No entanto, você está falando com um público não técnico, então certifique-se de simplificar conceitos complicados e
                                adotar um tom amigável e conversacional.
                                Se o trecho for irrelevante para a resposta, você pode ignorá-lo.

                                PERGUNTA: '{query}'
                                TRECHO: '{passage}'

                                RESPOSTA:
                            """).format(query=query, passage=escaped)

class RAGApplication:
    """Main RAG application class"""
    
    def __init__(self, api_key: str):
        self.gemini_client = GeminiClient(api_key)
        self.data_df = None
        
    def process_pdf(self, pdf_path: str):
        """Process PDF using Gemini's vision capabilities"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        # Convert PDF pages to images
        images = PDFProcessor.pdf_to_images(pdf_path)
        
        # Analyze each page
        page_contents = []
        page_analyses = []
        
        st.write("Analyzing PDF pages...")
        for i, image in enumerate(tqdm(images)):
            content = self.gemini_client.analyze_page(image)
            if content:
                # Store both the analysis and the content
                page_contents.append({
                    'page_number': i+1,
                    'content': content
                })
                page_analyses.append(content)
        
        if not page_analyses:
            raise ValueError("No content could be extracted from the PDF")
            
        # Create dataframe
        self.data_df = pd.DataFrame({
            'Original Content': page_contents,
            'Analysis': page_analyses
        })
        
        # Generate embeddings
        st.write("\nGenerating embeddings...")
        embeddings = []
        try:
            for text in tqdm(self.data_df['Analysis']):
                embeddings.append(self.gemini_client.create_embeddings(text))
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            time.sleep(10)
            
        _embeddings = []
        for embedding in embeddings:
            _embeddings.append(embedding.embeddings[0].values)
            
        self.data_df['Embeddings'] = _embeddings

    def answer_questions(self, questions: List[str]) -> List[Dict[str, str]]:
        """Responder a uma lista de perguntas usando os dados processados"""
        if self.data_df is None:
            raise ValueError("Por favor, processe um PDF primeiro usando process_pdf()")

        answers = []
        for question in questions:
            try:
                passage = self.gemini_client.find_best_passage(question, self.data_df)
                prompt = self.gemini_client.make_answer_prompt(question, passage)
                response = self.gemini_client.client.models.generate_content(
                    model=Config.MODEL_NAME,
                    contents=prompt
                )
                answers.append({
                    'question': question,
                    'answer': response.text,
                    'source': f"Página {passage['page']}\nConteúdo: {passage['content']}"
                })
            except Exception as e:
                print(f"Erro ao processar a pergunta '{question}': {e}")
                answers.append({
                    'question': question,
                    'answer': f"Erro ao gerar a resposta: {str(e)}",
                    'source': "Erro"
                })

        return answers

def main():
    # Carregar variáveis de ambiente
    load_dotenv()

    # Título da página
    st.set_page_config(page_title='🦜🔗 Ask the Doc App')
    st.title('🦜🔗 Ask the Doc App')

    # Obter a chave da API
    api_key = os.getenv('GOOGLE_API_KEY')
    alternative_names = ['GEMINI_API_KEY', 'GOOGLE_GEMINI_KEY', 'GEMINI_KEY']
    for name in alternative_names:
        if not api_key:
            api_key = os.getenv(name)
            if api_key:
                st.write(f"Chave da API encontrada em {name}")

    if not api_key:
        raise ValueError("Por favor, defina a variável de ambiente GOOGLE_API_KEY.")

    # Testar a chave da API
    try:
        test_client = genai.Client(api_key=api_key)
        test_response = test_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents="Olá, esta é uma mensagem de teste."
        )
        st.write("Chave da API está funcionando!", test_response.text)
    except Exception as e:
        st.write(f"Falha no teste da API: {e}")
        raise ValueError("Chave da API inválida.")

    # Formulário
    with st.form(key="stimy_form"):
        pdf_path = st.file_uploader("Carregar um arquivo PDF", type=["pdf"])
        questions = st.text_input('Digite sua pergunta:', placeholder='Por favor, forneça um breve resumo.')
        submit_button = st.form_submit_button(label="Enviar")

    if submit_button and pdf_path and questions:
        try:
            # Salvar o PDF carregado em um arquivo temporário
            temp_pdf_path = f"temp_{pdf_path.name}"
            with open(temp_pdf_path, "wb") as f:
                f.write(pdf_path.getbuffer())

            # Inicializar a aplicação
            app = RAGApplication(api_key)

            # Processar o PDF e responder às perguntas
            st.write(f"Processando PDF: {pdf_path.name}")
            with st.spinner("Pensando..."):
                app.process_pdf(temp_pdf_path)
                answers = app.answer_questions(questions)

            # Exibir as respostas
            for result in answers:
                st.write(f"Pergunta: {result['question']}")
                st.write(f"Resposta: {result['answer']}")
                st.write(f"Fonte: {result['source']}")
                st.write("-" * 80)
        except Exception as e:
            st.write(f"Ocorreu um erro: {e}")

if __name__ == "__main__":
    main()