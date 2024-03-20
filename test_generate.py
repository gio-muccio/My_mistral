"""
Si verifica l'equivalenza dei logit tra 
- utilizzo di tecniche di chunking + sliding window
- mancato utilizzo della tecnica di chunking (ma pur sempre sliding window)

Cioè si verifica se l'utilizzo di chunking e sliding window
per gestire lunghe sequenze di testo influisce o meno
sulla distribuzione dei logits e quindi di conseguenza sulla
qualità di generazione del testo.
"""

import torch
from typing import List

from mistral.model import ModelArgs, Transformer
from main import generate

# Classe definita/progettata per simulare un tokenizer di debug
# Fornisce metodi per codificare e decodificare sequenze di testo
class DebugTokenizer:
    @property
    # identificatore per il token di inizio sequenza
    def bos_id(self) -> int:
        return 0

    @property
    # identificatore per il token di fine sequenza
    def eos_id(self) -> int:
        return 1

    @property
    # identificatore per il token di padding
    def pad_id(self) -> int:
        return -1

    # input: stringa 's', boolean 'bos' che indica se includere il token di inizio sequenza
    # output: lista di interi
    def encode(self, s: str, bos: bool = True) -> List[int]:
        assert isinstance(s, str)
        t = [int(x) for x in s.split()]
        if bos:
            t = [self.bos_id, *t]
        return t
    """
    La stringa di input 's' viene suddivisa in token utilizzando 
    lo spazio come delimitatore e convertita in una lista di interi.
    Se il parametro 'bos' è True, viene aggiunto il token di 
    inizio sequenza all'inizio della lista.
    La lista di interi risultante viene restituita.
    """

    # input: lista di interi
    # output: stringa
    def decode(self, t: List[int]) -> str:
        return " ".join([str(x) for x in t])
    """
    Metodo che prende in input una lista di interi 't' e la converte 
    in una stringa concatenando i token separati da spazi.
    Viene quindi restituita la stringa risultante.
    """

# -> generare testo SENZA l'utilizzo della tecnica di chunking
# vengono eseguiti test per verificare che il modello generi correttamente
# il testo in base alle sequenze di input
def test_generation():
    torch.manual_seed(42)

    sequences = ["1 2 3 4 5 6 7", "0 1 2", "12 13 14", "2 4 34"]
    args = ModelArgs(
        dim=512,
        n_layers=1,
        head_dim=128,
        hidden_dim=2048,
        n_heads=4,
        n_kv_heads=2,
        sliding_window=3,
        norm_eps=1e-5,
        vocab_size=32_000,
        max_batch_size=len(sequences),
    )
    model = Transformer(args).to("cuda", dtype=torch.float32)
    tokenizer = DebugTokenizer()

    # Generazione delle sequenze di testo SENZA chuncking
    # for attempt in range(10):
    toks, all_logprobs_old = generate(sequences, model, tokenizer, max_tokens=7)
    toks = [" ".join(r.split(" ")[1:]) for r in toks] # Remove BOS
    generated, all_logprobs_new = generate(toks, model, tokenizer, max_tokens=0)
    assert generated == []
    
    # Verify that logprobs are the same
    assert len(sequences) == len(all_logprobs_old) == len(all_logprobs_new)
    for lp_old, lp_new in zip(all_logprobs_old, all_logprobs_new):
        assert all([abs(x - y) < 1e-5 for x, y in zip(lp_old, lp_new)]), f"\n{lp_old}\n{lp_new}"

    print("All tests passed.")


# -> generare testo CON l'utilizzo della tecnica di chunking
# vengono eseguiti test per verificare che il modello generi correttamente
# il testo in base alle sequenze di input utilizzando la tecnica
def test_chunks():
    torch.manual_seed(42)

    sequences = [" ".join([str(i) for i in range(7)]), " ".join([str(i) for i in range(9, 0, -1)])]
    # sequences = ["0 1 2 3 4 5 6", "9 8 7 6 5 4 3 2 1"]
    args = ModelArgs(
        dim=512,
        n_layers=1,
        head_dim=128,
        hidden_dim=2048,
        n_heads=4,
        n_kv_heads=2,
        sliding_window=4,
        norm_eps=1e-5,
        vocab_size=32_000,
        max_batch_size=3,
    )
    model = Transformer(args).to("cuda", dtype=torch.float32)
    tokenizer = DebugTokenizer()

    # Generazione delle sequenze i testo CON chunking
    # for attempt in range(10):
    toks, all_logprobs_old = generate(sequences, model, tokenizer, max_tokens=8)
    toks = [" ".join(r.split(" ")[1:]) for r in toks] # Remove BOS
    # Generazione e verifica di nuove sequenze di testo con chunking tramite specifica 'chunk_size'.
    generated, all_logprobs_new = generate(toks, model, tokenizer, max_tokens=0, chunk_size=5)
    assert len(generated) == 0

    for lp_old, lp_new in zip(all_logprobs_old, all_logprobs_new):
        assert all([abs(x - y) < 1e-5 for x, y in zip(lp_old, lp_new)]), f"\n{lp_old}\n{lp_new}"
    

if __name__ == "__main__":
    test_generation()
    test_chunks()


"""
Esecuzione delle due funzioni di test:  `test_generation()` e `test_chunks()`.
Entrambe progettate per verificare il corretto funzionamento della generazione di testo.

1. `DebugTokenizer`:
   - Questa classe simula un tokenizer di debug per la generazione di testo. 
   Essa fornisce metodi per codificare e decodificare sequenze di testo 
   utilizzando un elenco di token.

2. Funzione `test_generation()`:
   - Questa funzione testa la generazione di testo utilizzando il modello Mistral ***SENZA*** utilizzare la tecnica di chunking.
   - Viene definita una lista di sequenze di testo da generare.
   - Viene inizializzato un'istanza del modello Mistral con gli argomenti specificati.
   - Viene chiamata la funzione `generate()` per generare le sequenze di testo utilizzando il modello. 
     La variabile `max_tokens` specifica il numero massimo di token da generare per ciascuna sequenza.
   - Viene verificato che le sequenze generate siano vuote e che le log-probabilità delle sequenze generate 
     siano gli stessi delle log-probabilità delle sequenze originali. 
     Questo assicura che il modello generi lo stesso output ogni volta che viene eseguito con lo stesso input.

3. Funzione `test_chunks()`:
   - Questa funzione testa la generazione di testo utilizzando il modello Mistral ***CON*** la tecnica di chunking abilitata.
   - Funziona in modo simile alla funzione `test_generation()`, ma questa volta viene utilizzata, appunto, la tecnica di chunking 
     per gestire sequenze di testo più lunghe.
   - Viene verificato che le sequenze generate siano vuote e che le log-probabilità delle sequenze generate 
     siano gli stessi delle log-probabilità delle sequenze originali, confermando che la tecnica di chunking 
     non influisce sul risultato della generazione di testo.

In entrambi i "test", viene utilizzato un modello Mistral preaddestrato per generare le sequenze di testo, 
e vengono confrontati i risultati ottenuti con le sequenze originali per verificare la correttezza del modello 
e delle tecniche di generazione utilizzate.

Quando tale file viene lanciatod direttamente, allora vengono eseguiti i due test. Se tutti i test passano senza errori,
viene stampato il messaggio "All tests passed.". 
Questo indica che il modello sta funzionando correttamente secondo le aspettative.
"""