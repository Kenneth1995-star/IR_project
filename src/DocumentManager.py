class DocumentManager:
    def __init__(self):
        self.next_id = 0
        self.path_to_id = dict()
        self.deleted_docs = set()
        self.documents = list()

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
        with open(filepath, "r", encoding="utf8") as file:
            doc_content = file.read()
        return doc_id, doc_content

    def read_document_from_id(self, doc_id: int) -> str:
        with open(filepath, "r", encoding="utf8") as file:
            return file.read()