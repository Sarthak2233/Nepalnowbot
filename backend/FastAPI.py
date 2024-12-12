import os
import asyncio
import aiofiles
import threading
from threading import Thread
from datetime import datetime
from fastapi import FastAPI, HTTPException, requests
from fastapi.responses import RedirectResponse
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from scrape import extract_all_links_and_content, extract_anchor_tags, split_dom_content
from parse import create_summary_chain, create_qa_chain, question_generator
from storage import save_to_storage, get_from_storage

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust this to your frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define a Pydantic model for request validation
class QueryRequest(BaseModel):
    user_query: str

# File paths for storing data
SCRAPE_FILE_PATH = "ScrappedData/scripts/nepalnow.travel.txt"
REFINED_FILE_PATH = "ScrappedData/scripts/refined_content.txt"
ANCHOR_TAGS_FILE_PATH = "ScrappedData/scripts/anchor_tags.txt"
HARDCODED_URL = "https://nepalnow.travel"

users = {
    'name': 'Sarthak Basnet',
    'email' : 'sarthak.basnet5@gmail.com',
    'password': 'asdfghjkl'
}

async def load_from_file(file_path):
    if not os.path.exists(file_path):
        return None
    async with aiofiles.open(file_path, "r", encoding="utf-8") as file:
        return await file.read()
async def scrape_and_store_job():
    """
    Scrape the hardcoded URL and store the results automatically.

    """
    print(f"Starting scraping job at {datetime.now()}")
    async def save_to_file(file_path, content):
        async with aiofiles.open(file_path, "w", encoding="utf-8") as file:
            await file.write(content)

    print(f"Starting scraping job at {datetime.now()}")
    try:
        # Scraping logic here
        print(f"Scraping job completed successfully at {datetime.now()}")
        dom_content = []
        links = []
        # extracts = await asyncio.to_thread(await extract_all_links_and_content(HARDCODED_URL))
        # print("Extracts:", extracts)
        extracts = await extract_all_links_and_content(HARDCODED_URL)
        for url, content in extracts.items():
            print(f"URL: {url}")
            print(f"HTML Length: {len(content['html'])}")
            print(f"Body Content:\n{content['body']}\n")
            dom_content.append(content['body'])

        dom_contents = "".join(dom_content)
        # print("Dom_contents", dom_contents)
        links = await extract_anchor_tags(HARDCODED_URL)
        anchor_tags = "".join(links)

        await save_to_file(SCRAPE_FILE_PATH, dom_contents)
        await save_to_file(ANCHOR_TAGS_FILE_PATH, anchor_tags)
        await save_to_storage("dom_content", dom_contents)
        await save_to_storage("anchor_tags", anchor_tags)
        print(f"Scraping job completed successfully at {HARDCODED_URL}")
    except Exception as e:
        print(f"Error in scheduled scraping job: {e}")

async def generate_summary_job():
    def save_to_file(file_path, content):
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)

    """
    Automatically generate summaries from the scraped data.
    """
    print(f"Making Summary job at {datetime.now()}")
    dom_content = await load_from_file(SCRAPE_FILE_PATH)  # Await the file load
    if not dom_content:
        print("No DOM content available for summarization.")
        return

    try:
        with open(SCRAPE_FILE_PATH, 'r', encoding='utf-8') as file:
            contents = file.read()
            chunks = split_dom_content(contents)
            summaries = []

            for chunk in chunks:
                print('Processing chunk of size:', len(chunk))
                summary = create_summary_chain(chunk)
                summaries.append(summary)

            refined_contents = "\n".join(summaries)
            save_to_file(REFINED_FILE_PATH, refined_contents)
            save_to_storage("refined_content", refined_contents)
            print("Summary generated and stored successfully!")

    except Exception as e:
        print(f"Error during summary generation: {e}")


# Initialize and configure the scheduler
scheduler = AsyncIOScheduler()

@app.on_event("shutdown")
def shutdown_event():
    """
    Shutdown the scheduler when the application stops.
    """
    scheduler.shutdown()

@app.get("/")
def read_root():
    name = users['name']

    return {'response':"response",
            "message": "FastAPI is running. Scraping every hour!",
            "name": name}

first_run_completed = False

def run_scheduler():
    try:
        scheduler.start()  # Start the scheduler
        print("Scheduler started successfully.")
    except Exception as e:
        print(f"Error starting scheduler: {e}")


@app.on_event("startup")
async def startup_event():
    """
    Perform initial scraping and summary generation when the server starts.
    """
    print("Starting FastAPI server: Initializing scraping and summarization...")
    refined_content = await load_from_file(REFINED_FILE_PATH)

    # if refined_content:
    #     print("Loaded previous summary from file.")
    # else:
    #     print("No previous summary found. Performing initial scraping and summarization.")
    #     await scrape_and_store_job()  # Perform scraping
    #     await generate_summary_job()
    global first_run_completed
    if not first_run_completed:
        await asyncio.gather(
            scrape_and_store_job(),
            generate_summary_job()
        )
        first_run_completed = True

        # Start scheduler for periodic updates
    scheduler.add_job(scrape_and_store_job, "interval", hours=1, replace_existing=True)  # Run every hour
    scheduler.add_job(generate_summary_job, "interval", hours=1.2, replace_existing=True)  # Adjust start time as needed
    scheduler_thread = Thread(target=run_scheduler)
    scheduler_thread.daemon = True  # Ensures the thread exits when the main program does
    scheduler_thread.start()


async def autostart_logic(context, question):
    """
    Core logic for generating questions related to the context.
    """
    try:
        # Generate questions using LangChain
        questions = question_generator(context, question)

        # Format questions as buttons
        buttons = [{"question": q, "action": f"/query?user_query={q}"} for q in questions]
        return buttons
    except Exception as e:
        raise Exception(f"Failed to generate questions: {str(e)}")


@app.post("/autostart")
async def autostart(question: str):
    """
    Generate a list of questions related to the context and return as clickable buttons.
    """
    try:
        # Load context
        refined_content = get_from_storage("refined_content")
        if not refined_content:
            raise HTTPException(status_code=404, detail="Context not available")

        # Use the shared autostart logic
        questions = await autostart_logic(refined_content, question)
        return {"buttons": questions}
    except Exception as e:
        return {"error": f"Failed to generate questions: {str(e)}"}


@app.post("/query")
async def query_chatbot(request: QueryRequest):
    """
    Query the chatbot and ensure related questions are dynamically generated with /autostart.
    """
    try:
        user_query = request.user_query  # Access the user query from the request
        if not user_query:
            raise HTTPException(status_code=400, detail="Query cannot be empty.")
        # Load context from storage
        refined_content = await load_from_file(REFINED_FILE_PATH)
        scrapped_content = await load_from_file(ANCHOR_TAGS_FILE_PATH)
        if not refined_content:
            raise HTTPException(status_code=404, detail="Context not available")

        # Process the user's query and fetch an answer
        result = create_qa_chain(refined_content, user_query, scrapped_content )

        # Trigger the autostart logic to generate related questions
        questions = await autostart_logic(refined_content, user_query)

        return {
            "response": result,
            "related_questions": questions  # Include related questions as part of the response
        }
    except Exception as e:
        return {"error": f"Failed to process query: {str(e)}"}

