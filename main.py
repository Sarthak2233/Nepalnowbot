import os
import streamlit as st
from dotenv import load_dotenv
from backend.scrape import (
    extract_all_links_and_content,
    split_dom_content,
    extract_anchor_tags
)
load_dotenv()

from backend.parse import create_summary_chain, create_qa_chain

# Streamlit UI
# dom_content = []
# st.title("Chat About Any website")
#
# url = st.text_area("Enter the url of your choice.")
#
# if st.button("Generate Contents of the Entered website:"):
#     st.write("Scrapped.")
#
#         # Scrape the website
#     extracts = scrape_website(url)
#     file_path = f"E:/Projects/ScrapProject/ScrappedData/scripts/{url.split('.')[0]}.txt"
#     print("From Dom content",extracts)
#     # for url, content in extracts.items():
#     #     print(f"URL: {url}")
#     #     print(f"HTML Length: {len(content['html'])}")
#     #     print(f"Body Content:\n{content['body']}\n")
#     #     dom_content.append(content['body'])
#     # dom_contents = "".join(dom_content)
#     body_content = extract_body_content(extracts)
#     dom_contents = clean_body_content(body_content)
#     print("Dom_contents",dom_contents)
#
#     with open(file_path, "w", encoding="utf-8") as file:
#         file.write(dom_contents)
#     print(f"DOM content written to {file_path}")
#
#         # Store the DOM content in Streamlit session state
#     st.session_state.dom_contents = dom_contents
#
#     print("Cleaned content: ",dom_contents)
#         # Display the DOM content in an expandable text box
#     with st.expander("View DOM Content"):
#         st.text_area("DOM Content", st.session_state.dom_contents, height=300)
#     file_path = rf"E:/Projects/ScrapProject/ScrappedData/scripts/{url.split('.')[0]}.txt"
#     print(os.path.exists(file_path))
#     if os.path.exists(file_path):
#         with open(file_path, 'r') as file:
#             contents = file.read()
#         chunks = split_dom_content(contents)
#         print("Chunks: .....................",chunks)
#         try:
#             summaries = []
#             for chunk in chunks:
#                 summary = create_summary_chain(chunk, url)
#                 summaries.append(summary)
#             refined_contents = "\n".join(summaries)
#             st.session_state.refined_contents = refined_contents
#             with st.expander("Refined Summary"):
#                 st.text_area("Refined Text", st.session_state.refined_contents, height=1000)
#         except Exception as e:
#             print("Exception has occured", e)
#
# if "refined_contents" in st.session_state:
#     parse_description = st.text_area("Generate Answer")
#     if st.button("ANSWER"):
#         if parse_description:
#             st.write("Generated Answer.")
#
#             # Parse the content with Google genai
#             dom_chunks = split_dom_content(st.session_state.refined_contents)
#             print(f"DOM_Chunks--:{dom_chunks}")
#             parsed_result = question_answer_chain(dom_chunks, parse_description)
#             st.write(deduplicate_output(parsed_result))
#
#             if parsed_result:
#                 with st.container():
#                     st.markdown(
#                         """
#                                     <style>
#                                     .small-container {
#                                         position: fixed;
#                                         top: 10px;
#                                         left: 10px;
#                                         background-color: #f0f0f0;
#                                         padding: 10px;
#                                         border-radius: 5px;
#                                         box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
#                                         z-index: 1000;
#                                         max-width: 300px;
#                                     }
#                                     </style>
#                                     """,
#                         unsafe_allow_html=True,
#                         )
#                     st.markdown('<div class="small-container">', unsafe_allow_html=True)
#                     st.subheader("Token Usage")
#                     for doc, tokens in token_usage.items():
#                         st.write(f"**{doc}:**")
#                         st.write(f"- Input Tokens: {tokens['input_tokens']}")
#                         st.write(f"- Output Tokens: {tokens['output_tokens']}")
#                         st.markdown('</div>', unsafe_allow_html=True)

# import pandas as pd
# import os
# from parse import parse_with_googlegenai, token_usage
# from scrape import (
#     extract_all_links_and_content,
#     split_dom_content,
#     deduplicate_output,
#     all_links_of_techaxis
# )
# url = 'techaxis.com.np'
# file_path = r'E:\Projects\ScrapProject\ScrappedData\scripts\ScrappedData.txt'
# dom_contents = []
# extracts = extract_all_links_and_content(url)
# for url, content in extracts.items():
#     print(f"URL: {url}")
#     print(f"HTML Length: {len(content['html'])}")
#     print(f"Body Content:\n{len(content['body'])}\n")
#     dom_contents.append(content['body'])

# contents = "".join(dom_contents)
# with open(file_path, "w", encoding="utf-8") as file:
#     file.write(contents)
# print(f"DOM content written to {file_path}")
#


# Initialize session state
if "dom_contents" not in st.session_state:
    st.session_state.dom_contents = ""

if "refined_contents" not in st.session_state:
    st.session_state. refined_contents = ""

if "parsed_result" not in st.session_state:
    st.session_state.parsed_result = ""

if "anchor_tags" not in st.session_state:
    st.session_state.anchor_tags = ""

# Title
st.title("ChitChat Informer")

# URL Input
base_url = 'https://nepalnow.travel'

if st.session_state.refined_contents and st.session_state.anchor_tags:
    parse_description = st.text_area("Enter your question:")

    if st.button("ANSWER"):
        if parse_description:
            try:
                dom_chunks = split_dom_content(st.session_state.refined_contents)
                parsed_result = create_qa_chain("".join(dom_chunks), parse_description, st.session_state.anchor_tags)
                st.session_state.parsed_result = parsed_result
                print(parsed_result)

                # Display the parsed result
                with st.container():
                    st.components.v1.html(parsed_result, height=3000)
            except Exception as e:
                st.error(f"Error: {e}")
else:
    # Generate Content Button
    if base_url:
        dom_content = []
        links = []
        extracts = extract_all_links_and_content(base_url)
        for url, content in extracts.items():
            print(f"URL: {url}")
            print(f"HTML Length: {len(content['html'])}")
            print(f"Body Content:\n{content['body']}\n")
            dom_content.append(content['body'])


        dom_contents = "".join(dom_content)
        # print("Dom_contents", dom_contents)
        links = extract_anchor_tags(base_url)
        anchor_tags = "".join(links)
        # print(type(anchor_tags))
        #
        with open(f"/backend/ScrappedData/scripts/nepalnow.travel.txt", "w", encoding="utf-8") as file:
            file.write(dom_contents)


        print(f"DOM content written to E:/Projects/ScrapProject/ScrappedData/scripts/nepalnow.travel.txt")

        # Save to session state
        st.session_state.dom_contents = dom_contents
        st.session_state.anchor_tags = anchor_tags

        file_path = rf"/backend/ScrappedData/scripts/nepalnow.travel.txt"
        print(os.path.exists(file_path))
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                contents = file.read()
                chunks = split_dom_content(contents)
                print("Chunks: .....................",chunks)
                try:
                    # summary = create_summary_chain(contents)
                    summaries = []
                    for chunk in chunks:
                        print('Chunk:' , len(chunk))
                        summary = create_summary_chain(chunk)
                        summaries.append(summary)
                    refined_contents = "\n".join(summaries) #summary
                    # Save refined content
                    st.session_state.refined_contents = refined_contents
                except Exception as e:
                    print("Exception has occured", e)

        st.success("Content generated successfully!")



    # Show DOM Content
        if st.session_state.dom_contents:
            with st.expander("View DOM Content"):
                st.text_area("DOM Content", st.session_state.dom_contents, height=300)

        # Show Refined Summary
        if st.session_state.refined_contents:
            with st.expander("View Refined Summary"):
                st.text_area("Refined Content", st.session_state.refined_contents, height=456)

        if st.session_state.anchor_tags:
            with st.expander("ALL the anchor tags"):
                st.text_area("Tags in the page:", st.session_state.anchor_tags, height=200)

        # Question Answering
        if st.session_state.refined_contents and st.session_state.anchor_tags:
            parse_description = st.text_area("Enter your question:")

            if st.button("ANSWER"):
                if parse_description:
                    try:
                        dom_chunks = split_dom_content(st.session_state.refined_contents)
                        parsed_result = create_qa_chain("".join(dom_chunks), parse_description, st.session_state.anchor_tags)
                        st.session_state.parsed_result = parsed_result
                        print(parsed_result)

                        # Display the parsed result
                        with st.container():
                            st.components.v1.html(parsed_result, height=3000)
                    except Exception as e:
                        st.error(f"Error: {e}")
