# 🤖 RAG PDF Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows you to interact with your PDF documents using natural language queries. Powered by Microsoft's BitNet model, this chatbot runs efficiently on CPUs, eliminating the need for costly GPUs.

## 🚀 Features

- **Local Execution**: Ensures data privacy and security.
- **CPU-Only Inference**: No GPU required.
- **Efficient Retrieval**: Utilizes FAISS for quick document chunk retrieval.
- **Interactive Interface**: Built with Streamlit for ease of use.

## 🛠️ Technologies Used

- [BitNet b1.58 2B4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
- [bitnet.cpp](https://github.com/microsoft/BitNet)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/)
- [Streamlit](https://streamlit.io/)

## 📦 Installation

- [if you are getting lots of error while installation DM me , I have working commands.] https://github.com/microsoft/BitNet

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name

2. **Create a virtual environment**:
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
    pip install -r requirements.txt

4. Download the BitNet model:
5. Run the application:

📁 Directory Structure
   
   your-repo-name/
├── app.py

├── models/
│   └── bitnet-model.gguf

├── rag-dataset/
│   └── your-pdfs.pdf

├── requirements.txt

└── README.md "

🧠 How It Works
PDF Processing: Upload PDFs which are then parsed and split into text chunks.

Embedding Generation: Each chunk is converted into embeddings using Hugging Face models.

Vector Store Creation: Embeddings are stored using FAISS for efficient retrieval.

Query Handling: User queries are embedded and compared against the vector store to find relevant chunks.

Response Generation: The BitNet model generates responses based on the retrieved context.

🔒 Security & Efficiency
Local Execution: All processes run locally, ensuring data privacy.

CPU Optimization: BitNet's 1.58-bit quantization allows efficient CPU inference.

Cost-Effective: Eliminates the need for expensive GPU resources.

🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📌 Additional Tips

- **Visuals**: https://drive.google.com/file/d/1d9HrsDeKg317vHh5CteI3tDIyPzzrzku/view?usp=sharing
- **Preview**: https://www.youtube.com/@amolm987     '**stay tuned for video**'
 

