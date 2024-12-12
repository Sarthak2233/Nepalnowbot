import re
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain import hub
from langchain.chains.llm import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.chains.question_answering import load_qa_chain
import os
import tiktoken
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, Runnable, RunnableMap, RunnableLambda
from concurrent.futures import ThreadPoolExecutor
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI, HarmCategory
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

# Load environment variables from .env file
# API_KEY = os.environ['OPENAI_API_KEY']
#
# if not API_KEY:
#     raise ValueError("GEMINI apikey not found. Please add it to your .env file.")

# # # Initialize the OpenAI model
token_usage = {}

# model = ChatOpenAI(
#     api_key=API_KEY,
#     temperature=0.5,
#     timeout=None,
#     model='gpt-4o-mini',
#     max_tokens=420,
# )
load_dotenv()

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    temperature=0.1,
    GOOGLE_API_KEY=os.environ['GEMINI_API_KEY'],
)
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    temperature=0.1,
    GOOGLE_API_KEY=os.environ['GEMINI_API_KEY'],
)


# embedding_model = OpenAIEmbeddings(
#     model="text-embedding-ada-002",
# )
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs).replace("\n", "")


def create_faiss_index(chunks):
    """
    Create a FAISS index from the given chunks.

    Args:
        chunks (list): List of text chunks.
        embedding_model: Embedding model to generate vector representations.

    Returns:
        FAISS: A FAISS index with the embedded chunks.
    """
    documents = [Document(page_content=chunk) for chunk in chunks]
    embeddings = embedding_model
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store
def create_summary_chain(text):

    model = ChatGoogleGenerativeAI(
        model='models/gemini-1.5-pro',
        temperature=0.5,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=os.environ['GEMINI_API_KEY'],
    )

    # parsed_results = []
    # vector_store = create_faiss_index(text)

    # Define the query for similarity search
    query = (
        f"Your task is to provide the long description provided by Nepalnow from the given context all written in the webiste"
        f"The text should contain all the relevent information, stories,news and events contained from the "
        f"provided context."
    )

    # Updated prompt with correct input variable names
    summary_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=("""You are a summarization expert. Use the following context to create a detailed, accurate, and well-structured summary. 
        Focus on extracting all relevant information present in the context,
        the context is from the website calledNepalNow and present it in an organized paragraph format.
        Remember to include all the details required for tourism and news about Nepal that the context has.
        Context: {context}
        Question: {question}

        Provide the output as a comprehensive summary. Avoid vague or generic responses.""")
    )
    summary_chain = LLMChain(
        llm=model,
        prompt=summary_prompt,
        output_parser=StrOutputParser()  # Ensures output is properly parsed
    )
    answer = summary_chain.run(
        {
            'context': text,
            'question': query,
        }
    )
    print('Your summary has been created: ',type(answer))
    return answer
    # Perform similarity search
    # print(f"Simalirity tracked.")
    # docs = vector_store.similarity_search_with_score(query, k=1)
    # print(f"Documents found: {docs}")
    # # Load the QA chain with updated prompt
    # summary_chain = load_qa_chain(model, chain_type="stuff", prompt=summary_prompt)
    #
    # # Extract relevant documents based on score threshold
    # relevant_docs = [doc for doc, score in docs if score >= 0.7]
    # print(f"Relevant documents: {len(relevant_docs)}")
    # # Parse the relevant chunks
    # if relevant_docs:
    #     with ThreadPoolExecutor() as executor:
    #         responses = list(
    #             executor.map(
    #                 lambda doc: summary_chain.run(input_documents=[doc], question=query),
    #                 relevant_docs
    #             )
    #         )
    #
    #         parsed_results.extend(
    #             response.strip("\n") if hasattr(response, "content") else str(response)
    #             for response in responses
    #         )
    #
    # return "\n".join(parsed_results)

class NormalAnswer(Runnable):
    def __init__(self, message):
        # self.llm = model
        self.llm = ChatGoogleGenerativeAI(
            model='models/gemini-1.5-pro',
            temperature=0.8,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=os.environ['GEMINI_API_KEY'],
            safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
                                 HarmBlockThreshold.BLOCK_NONE},
        )

        self.message = message

    def invoke(self, config: Any = None) -> str:
        '''
          Args humman_message: a dictionary containing a question from the qa chain
        '''
        print("Type of input from the Normal Answer invoke", type(self.message))

        message_template = PromptTemplate(
            input_variables = ['message'],
            template = 'You are a omniscient person and you can answer whatever the message is given to you about Nepal,its news, and traveling tips.'
                     "You are nepalNow."
                     "Don't add ```html  and ```  inside the generated content"
                     'You always give the answer in less token as possible.'
                     "Just give the brief overview of the message given to you. Try and be friendly."
                       'Message:  {message} '
                     """Your response must have the following properties:
                        - Return the DOM CONTENT ONLY
                        - Don't add ```html  and ```  inside the generated content
                        - Be enclosed within a <div> tag to clearly present the content.
                        - Bold important information using <b> tags.
                        - Maintain the structure, and include key details, without leaving out any critical points.
                        - Not content before initial <p> tag and after the last </p> tag.
                        - Only the proper HTML tags and structure inside the <p> tag should be present.
                        - Provide 'https://nepalnow.travel' in the content at the last part of response in the <a> anchor tag(full link) When user click the links it opens in new tab
                        - Don't add (```html) and (```)  inside the generated content 
                        """

        )
        answer = LLMChain(llm=self.llm, prompt=message_template, output_parser=StrOutputParser(),)

        response = answer.run({'message' : self.message})
        return response


class HyperlinkExtractor(Runnable):
    def __init__(self,  question):
        """
        Initialize the extractor with an LLM for relevance filtering.
        Args:
            llm: An instance of a language model (e.g., OpenAI's GPT).
        """
        self.llm = ChatGoogleGenerativeAI(
            model='models/gemini-1.5-pro',
            temperature=0.3,
            max_tokens=250,
            timeout=None,
            max_retries=2,
            api_key=os.environ['GEMINI_API_KEY'],
            safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
                                 HarmBlockThreshold.BLOCK_NONE},
        )
        # Use the LLM for filtering relevant links
        self.question = question

    def invoke(self, input_text: Dict, config: Any = None) -> List[str]:
        """
        Extracts hyperlinks from the input text and filters them using the LLM.
        Args:
            input_text: A dictionary containing 'scrapped_content' and 'question'.
        """
        print("Type of the input_text from invoke", type(input_text))
        context = input_text['context']
        text = input_text['scrapped_content']
        question = input_text['question']

        print(type(question))
        print("\n*8 \n Question from HyperlinkExtractorTool :", question)
        # Step 1: Extract all links from the scrapped content
        url_pattern = r"https?://\S+"
        all_links = re.findall(url_pattern, text)
        print("All the scrapped Links:", all_links)

        # Step 2: Ask the LLM to select the most relevant links
        relevant_links = self.filter_links_with_llm(all_links, self.question)
        print()

        return relevant_links

    def filter_links_with_llm(self, links: List[str], question: str) -> List[str]:
        """
        Use the LLM to filter and rank the most relevant links.
        Args:
            links: List of all extracted links.
            question: The user's question.
        Returns:
            A list of the top 3 relevant links.
        """
        # Prepare a prompt for the LLM
        formatted_links = "\n".join(f"- {link}" for link in links)
        prompt = f"""You are an expert link extractor. Your task is to find links from the given webpage content that are relevant to the question. Minimize your response tokens and adhere to the following rules:

                    - Provide your response as valid HTML enclosed within a single <div> tag.
                    - Don't add ```html   inside the generated content
                    - Bold important details using <b> tags.
                    - Recommend top links from {formatted_links} based on relevance.
                    - If no links match, include a fallback link: <a href='https://nepalnow.travel' target='_blank'>https://nepalnow.travel</a>.
                    - Ensure all links open in a new tab using target='_blank'.
                    - Do not include any text or code outside the <div> tag.
                    - Respond only with proper HTML tags."
                    
                    Question: {question}

                 """

        link_formatting_prompt = PromptTemplate(
            input_variables=["question"],
            template=prompt
        )
        # Use the LLM to generate a response
        link_generating_chain = LLMChain(
            llm=self.llm,
            prompt=link_formatting_prompt,
            output_parser=StrOutputParser()  # Ensures output is properly parsed
        )
        response = link_generating_chain.run({
            'question': question,
        })
        # Extract the top 3 links from the LLM's response (assuming each link is on a new line)
        print("Response from Hyperlink extractor: ", type(response))
        top_links = response.strip().split("\n")[:3]
        print(type(top_links))
        print("Top links,", top_links)
        return top_links


def simple_answer_chain(question):
    print("Simple answer chain invoked.")
    normal_answer = NormalAnswer(question)
    print('simple answer chain', type(normal_answer))

    return normal_answer.invoke()

def is_question_in_context(context: str, question: str) -> bool:
    contexts = [Document(page_content=context)]
    vector_store = FAISS.from_documents(contexts, embedding_model)
    answers = vector_store.similarity_search_with_score(question, k=2)
    print(answers[0][1])
    if 0.4 <= answers[0][1] <= 1:
        return True
    else:
        return False


def create_qa_chain(docs, question, scrapped_content):
    """
    :param docs: are the context
    :param question: question from the user
    :param scrapped_content: all the links
    :return: the answer to the question in a string format
    """
    print('From QA chain: ', type(scrapped_content))

    def combine_results(inputs):
        """Combine QA answer and hyperlinks."""
        answer = inputs["answer"]
        links = inputs["links"]
        print(links)
        if links:
            print("From combine result:",links)
            answer += "\n".join(links)
        return answer

    model = GoogleGenerativeAI(
        model='models/gemini-1.5-pro',
        temperature=0.3,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=os.environ['GEMINI_API_KEY'],
        safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
                         HarmBlockThreshold.BLOCK_NONE},
    )

    prompt_template = PromptTemplate(
        input_variables=["context", "question", "scrapped_content"],
        template=(
            "You are a intelligent customer service chatbot for nepalnow and your job is to give answer about question. "
            """"Your job is to answer all the question a customer might ask about. 
            Your answer must be exact to the question. 
            Provide the answer in a simple paragraph that the customer 
            can read. You are not alllowed to provide any links in the response.
            "If the question is broad or ambiguous, prioritize 
            providing the most relevant information first. And don't try to give negative statements in your response.
            "Text: {context}. "
            "link: {scrapped_content}"
            "Question: {question}"""
            """Your response must have the following properties:
                        - Return the DOM CONTENT ONLY
                        - Don't add ```html  and ```  inside the generated content
                        - Be enclosed within a <div> tag to clearly present the content.
                        - Bold important information using <b> tags.
                        - Maintain the structure, and include key details, without leaving out any critical points.
                        - Not content before initial <p> tag and after the last </p> tag.
                        - Only the proper HTML tags and structure inside the <p> tag should be present.
                        - Provide 'https://nepalnow.travel' in the content at the last part of response in the <a> anchor tag(full link) When user click the links it opens in new tab
                        - Don't add (```html) and (```)  inside the generated content      
            
            """

        )
    )
    if is_question_in_context(docs, question):
        qa_chain = prompt_template | model | StrOutputParser()
        print("Print QA chain ", qa_chain)
        parallel_chain = RunnableMap({
            'answer': qa_chain,
            'links': HyperlinkExtractor(question),
        })
        print("\n*7 \n Parallel chain", parallel_chain)

        full_chain = parallel_chain | RunnableLambda(combine_results) | StrOutputParser()
        answer = full_chain.invoke(
            {
                'context': docs,
                'question': question,
                'scrapped_content': scrapped_content
            }
        )

    else:
        answer = simple_answer_chain(question)


    return answer.strip("\n") if hasattr(answer, "content") else str(answer)

def question_generator(context, question):
    prompt = PromptTemplate(
        input_variables=["context","query"],
        template="Based on the following context, {context} "
                 "generate a list of 3 short questions related to query: {query}"
                 "Your response must be question only with out number listing and no"
                 "description of the query realating to context"
    )
    llm = ChatGoogleGenerativeAI(
        model='models/gemini-1.5-pro',
        temperature=0.3,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=os.environ['GEMINI_API_KEY'],
        safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
                         HarmBlockThreshold.BLOCK_NONE},
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    # Generate the questions
    questions = chain.run({"context" : context, 'query': question})
    return questions.split("\n")











# Define the prompt template
# template = (
#     "You are friendly and is expert in extracting specific information from the following text content: {context}. "
#     "Please follow these instructions carefully: \n\n"
#     "1. **Extract Information:** Make your own answer that directly matches the provided Question: {question}.  "
#     "2. **No Extra Content:** Do not include any additional text, comments, or explanations in your response. "
#     "3. **No Information:** if no information is found then continue returning whitespace"
#     "4. **Direct Data Only:** Your output should contain only the data that is explicitly requested, with no other text."
#     "5. **Out of Context Response:** If the information is not present in the context, use your own knowledge to generate an answer to the question. Ensure your response is accurate and helpful.\n"
#     "Note that dates also be an inclusion for some questions taken from the context."
# )
#
#
# def parse_with_googlegenai(dom_chunks, question, top_k=1):
#     """
#     Parses content using LangChain and FAISS for similarity search while tracking token usage.
#
#     Args:
#         dom_chunks (list): List of DOM content chunks to parse.
#         parse_description (str): Description of the parsing task.
#         top_k (int): Number of most relevant chunks to retrieve.
#
#     Returns:
#         str: string of parsed results.
#     """
#     # Initialize a dictionary to store token usage
#     # Create FAISS index
#     vector_store = create_faiss_index(dom_chunks)
#
#     # Build the chain
#     prompt = ChatPromptTemplate.from_template(template)
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#
#     # Tokenizer for tracking
#     tokenizer = tiktoken.encoding_for_model("gpt-4o-mni")  # Replace with appropriate model if needed
#
#     # Perform similarity search
#     query = question
#     docs = vector_store.similarity_search_with_score(query, k=top_k)
#     # Check if any document is relevant
#     relevant_docs = [doc for doc, score in docs if score >= 0.3]
#     # Parse the relevant chunks
#     if relevant_docs:
#         parsed_results = []
#         for i, (doc, score) in enumerate(docs, start=1):
#             if score >= 0.3:
#                 # Tokenize input document and query
#                 input_text = doc.page_content + "\n\n" + question
#                 input_tokens = len(tokenizer.encode(input_text))
#
#                 # Run the chain
#                 response = chain.run(
#                     input_documents=[doc],
#                     question=question
#                 )
#
#                 # Tokenize output response
#                 output_tokens = len(tokenizer.encode(response if hasattr(response, 'content') else str(response)))
#
#                 # Store token counts in the dictionary
#                 token_usage[f"Document {i}"] = {
#                     "input_tokens": input_tokens,
#                     "output_tokens": output_tokens
#                 }
#
#                 # print(f"Parsed document: {i} of {top_k}")
#                 print(response)
#                 print(type(response))
#                 parsed_results.append(response.strip("\n") if hasattr(response, 'content') else str(response))
#
#             return "\n".join(parsed_results)
#
#
# def question_answer_chain(dom_chunks, question):
#     tokenizer = tiktoken.encoding_for_model('gpt-4o-mini')
#     input_text = "".join(dom_chunks) + "" + question
#     input_token = len(tokenizer.encode(input_text))
#     vectorstore = create_faiss_index(dom_chunks)
#     retriever = vectorstore.as_retriever()
#     prompt = PromptTemplate(
#         input_variables=["context", "question"],
#         template=(
#             f"You are a intelligent customer service agent and your job is to get all the information about the gieven context. "
#             f""""Your job is to answer all the information a potential customer might ask about. This includes:
#                 Company Overview: What does the company do? What products or services do they offer?
#                 Contact Information: How can customers reach the company (e.g., email, phone, chat)?
#                 Pricing and Packages: What are the pricing plans, subscription options, or payment methods?
#                 Policies: What are the return, refund, or cancellation policies? Are there shipping and delivery details?
#                 FAQs: What are the most common questions and their answers provided on the website?
#                 Numerical Datas: Any thing relating to number and the company
#                 Special Features: What makes the company or its offerings unique? Are there any ongoing promotions, discounts, or special services?
#                 Additional Information: Are there any testimonials, students, case studies, or blog content that might be helpful to customers?
#                 Using the website's content,give response that could help the customer interacting via a chat service.
#                 Ensure the tone is professional, friendly, and clear."
#                 """
#             "Text: {context}. "
#             "Question: {question}"
#         )
#     )
#     chain = (
#             {"context": retriever | format_docs, "question": RunnablePassthrough()}
#             | prompt
#             | model
#             | StrOutputParser()
#     )
#     answer = chain.invoke(question)
#     output_token = len(tokenizer.encode(answer))
#
#     token_usage['input'] = input_token
#     token_usage['output'] = output_token
#     return answer.strip("\n") if hasattr(answer, "content") else str(answer)
