This is made with react, fastapi and langchain that uses the googel gemini model.

The react api posts the data to the prot in fast api with a query and it answers the response in the div format sent from the langchain. There are bunch of chains and on the basis of context measure range refering from the generated summary is how the model responds about the query asked. There is a stand alone class for the links to extract by the model on the basis of the context and the query. The Fast api is supposed to auto start every hour and call the functionalities from scrape and make the summary and stores.
