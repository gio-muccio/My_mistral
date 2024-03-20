"""
Implementazione di un layer di Mixture of Experts (MoE)
Quindi si ha la combinazione degli output degli experts selezionati.
"""

import dataclasses
from typing import List

import torch
import torch.nn.functional as F
from simple_parsing.helpers import Serializable
from torch import nn


@dataclasses.dataclass
class MoeArgs(Serializable):
    num_experts: int
    num_experts_per_tok: int


class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, moe_args: MoeArgs):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.args = moe_args

    def forward(self, inputs: torch.Tensor):
        # Per ogni token dell'input, genera `num_experts` logit che saranno utilizzati per indicare quale esperto utilizzare.
        gate_logits = self.gate(inputs)
        # Per ogni token->logit, seleziona i primi `num_experts_per_tok` esperti 
        weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok)
        # Applica la softmax ai logit DOPO aver selezionato i migliori/top k
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        # Creazione tensore di zeri della stessa dim di inputs che servirà per memorizzare l'output aggregato degli esperti
        results = torch.zeros_like(inputs)
        # Segue codice per combinare i singoli risultati in un unico risultato (weighted sum)
        # Iterando su ogni esperto, si restiuisce una tupla contenente l'indice dell'esperto 'i' e l'esperto stesso 'expert'
        for i, expert in enumerate(self.experts):
            # Per ogni esperto, si seleziona a quale token verrà applicato/assegnato.
            # In particolare, vengono identificati gli indici dei batch 'batch_idx' e l'indice dell'esperto all'interno dei batch 'nth_expert'
            # che sono stati appunto selezionati per l'esperto corrente
            batch_idx, nth_expert = torch.where(selected_experts == i)
            # Applica l'esperto ai token selezionati nel batch ponderandolo per i pesi ("ex-logit" dopo la softmax) calcolati sopra.
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                inputs[batch_idx]
            )
        return results
    
    """
    In pratica, 'results[batch_idx]' viene aggiornato aggiungendo il prodotto tra
    - il peso di gating (dal gate) ottenuto da 'weights[batch_idx, nth_expert, None]'
    e
    - l'output dell'esperto corrispondente all'input del batch ottenuto da 'expert(inputs[batch_idx]'
    
    Una volta avvenuta tale combinazione degli output degli experts per tutti i batch, si restituisce l'unico
    risultato della funzione 'forward', ossia si restituisce 'results'.
    Quindi 'results' contiene l'output aggregato degli esperti, dove ciascun output è stato pesato in base
    ai pesi calcolati dal gate.
    """
