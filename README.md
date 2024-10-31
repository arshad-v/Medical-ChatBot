
# Medical Chatbot Using RAG (Retrieval-Augmented Generation)



This project presents a Medical Chatbot developed with a Retrieval-Augmented Generation (RAG) approach, designed to assist users in obtaining reliable medical information. The chatbot leverages RAG architecture to fetch relevant information and generate natural language responses, combining the strengths of retrieval-based and generative language models to deliver accurate and contextually relevant responses.



## Features

- Contextual Medical Query Handling: The chatbot processes natural language queries and provides contextually relevant responses.
- Knowledge Retrieval: Utilizes external medical knowledge sources to retrieve accurate information
- Generative Response: Employs a generative model to formulate responses in a conversational manner.
- Scalable Architecture: Can be extended to integrate with various medical data sources for enhanced accuracy.




## Installation
### Prerequisites
* Python 3.8+
* Required libraries can be found in `requirements.txt`. Install them using:

```bash
  pip install -r requirements.txt
```

### Setting Up Environment Variables
Add the required API keys and configurations in an `.env` file (template provided in the repository). 
## Usage
1. Run the RAG Model: Open `rag.ipynb` to initiate the retrieval and generation components of the chatbot.
2. Launch Chatbot: Use `chat.py` to start the chatbot interface.

To start the chatbot, run:

```bash
python chat.py
```
Then, input your medical queries directly into the console to receive responses.



## Project Structure

* chat.py: Main script for running the chatbot interface.
* rag.ipynb: Jupyter notebook implementing the Retrieval-Augmented Generation model.
* .env: Environment file for storing API keys and configurations.
* requirements.txt: List of required packages.
## How it Works

1. Retrieval Component: Searches through a predefined set of medical documents or databases to find relevant information.
2. Generation Component: Utilizes a generative language model to create a coherent response based on retrieved information.
3. Combined RAG Workflow: The chatbot first retrieves relevant documents, then generates a response based on the query and retrieved information, ensuring responses are both informative and contextually accurate.
## Contributing

Contributions are always welcome! Please follow steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a Pull Request.

## License

This project is licensed under the [MIT](https://choosealicense.com/licenses/mit/) License.

