import nltk

class NltkNer:
    def __init__(self):
        self.ensure_nltk_resource('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle')
        self.ensure_nltk_resource('punkt', 'tokenizers/punkt/english.pickle')
        self.ensure_nltk_resource('words', 'corpora/words')
        self.ensure_nltk_resource('maxent_ne_chunker_tab', 'chunkers/maxent_ne_chunker_tab/english_ace_multiclass/')

    def ensure_nltk_resource(self, resource_id, resource_path):
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_id)

    def extract(self, english_text: str):
        tokens = nltk.word_tokenize(english_text)  # разбивает текст на токены (слова)
        return nltk.pos_tag(tokens)  # правильно без lang=...

    def get_named_entities(self, english_text: str):
        named_entities = []
        for sent in nltk.sent_tokenize(english_text):
            tokens = nltk.word_tokenize(sent)
            tagged_tokens = nltk.pos_tag(tokens)
            chunked = nltk.ne_chunk(tagged_tokens)

            for chunk in chunked:
                if hasattr(chunk, 'label'):
                    entity_name = ' '.join(c[0] for c in chunk)
                    entity_type = chunk.label()
                    named_entities.append((entity_name, entity_type))
        return named_entities

