from langchain_community.document_loaders import CSVLoader, JSONLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def csv_load() -> None:
    loader = CSVLoader(
        file_path="tv_shows.csv", csv_args={"delimiter": ",", "quotechar": '"'}, metadata_columns=("title",)
    )
    # documents = loader.load()
    for document in loader.lazy_load():
        print(document)


def json_load() -> None:
    loader = JSONLoader(file_path="tv_shows.json", jq_schema=".[]", text_content=False)
    documents = loader.load()
    for document in documents:
        print(document)


def text_load() -> None:
    loader = TextLoader(file_path="article.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200, chunk_overlap=50, separators=["\n\n", "\n", " ", "", "?", "!", "."], length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        print("\n" + "=" * 20 + f"Chunk {index}" + "=" * 20)
        print(chunk)


def pdf_load() -> None:
    loader = PyPDFLoader(file_path="pypdfloader_example.pdf")
    documents = loader.load()
    for document in documents:
        print(document)


if __name__ == "__main__":
    csv_load()

    # json_load()

    # text_load()
    # pdf_load()
