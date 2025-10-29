class Lexicon:
    def __init__(self):
        self.next_id: int = 0
        self.term_to_id: dict[str, int] =  dict()
        self.terms: list[str] = list()
        # self.document_frequency: dict[str, int] = dict()
    
    def get_id(self, term: str) -> int:
        if term in self.term_to_id:
            return self.term_to_id[term]

        term_id = self.next_id
        self.term_to_id[term] = term_id
        self.next_id += 1

        self.terms.append(term)
        return term_id
    
    def get_term(self, term_id) -> str | None:
        if term_id > len(self.terms):
            return None
        return self.terms[term_id]

    def save(self):
        pass
        