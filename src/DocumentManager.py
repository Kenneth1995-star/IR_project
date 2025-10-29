class DocumentManager:
    def __init__(self):
        self.next_id = 0
        self.path_to_id = dict()
        self.documents = list()
        self.deleted_docs = set()

    def add_document(self, filepath) -> int:
        if filepath in self.path_to_id:
            if (doc_id := self.path_to_id[filepath]) in self.deleted_docs:
                self.deleted_docs.remove(doc_id)
            return doc_id

        doc_id = self.next_id
        self.next_id += 1
        self.path_to_id[filepath] = doc_id
        self.documents.append(filepath)
        return doc_id

    def delete_document(self, doc_id: int) -> None:
        self.deleted_docs.add(doc_id)

    def is_deleted(doc_id: int) -> bool:
        return doc_id in self.deleted_docs

    def read_document(self, filepath) -> tuple[int, str]:
        doc_id = self.add_document(filepath)
        with open(filepath, "r", encoding="utf8") as f:
            return doc_id, f.read()

    def read_document_from_id(self, doc_id: int) -> str:
        with open(self.documents[doc_id], "r") as f:
            return f.read()

    def read_document_stream(self, filestream):
        while filepath := next(filestream, None):
            doc_id = self.add_document(filepath)
            with open(filepath, "r") as f:
                for line in f:
                    yield line
    
    def read_document_stream_from_id(self, doc_id: int):
        with open(self.documents[doc_id], "r") as f:
            for line in f:
                yield line