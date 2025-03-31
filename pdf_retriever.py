from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


class PdfRetriever:
    """
    Charge un fichier PDF, segmente en morceaux de texte et récupère des documents pertinents.
    """
    def __init__(self, path: str, chunk_size: int = 500, chunk_overlap: int = 250, top_k: int = 10):
        """
        Initialise le chargeur de PDF et crée le retrieveur de documents.

        :param path: (str) Chemin du fichier PDF à traiter.
        :param chunk_size: (int) Nombre de caractères par chunk, par défaut 500.
        :param chunk_overlap: (int) Nombre de caractères qui se chevauchent entre les chunks, par défaut 250.
        :param top_k: (int) Nombre maximum de documents pertinents à récupérer lors d'une requête, par défaut 10.
        """
        self.path = path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        # Initialiser un modèle d'embedding local
        self.embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = self._process_pdf()

    def _process_pdf(self):
        """Charge le PDF, segmente le texte et crée un stockage vectoriel."""
        loader = PyMuPDFLoader(self.path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        doc_chunks = text_splitter.split_documents(documents)
        
        return Chroma.from_documents(documents=doc_chunks, embedding=self.embedding_function)  # Utilisation correcte

    def get_retriever(self):
        """Retourne un retrieveur de documents basé sur les embeddings."""
        return self.vectorstore.as_retriever(top_k=self.top_k)
