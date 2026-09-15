import os
import torch
import torch.nn as nn
try:
    from opt_einsum import contract
except ImportError:
    def contract(equation, *operands):
        return torch.einsum(equation, *operands)
import torch.nn.functional as F
from long_seq import process_long_input
from losses import ATLoss
from validate_fixed_mean import validate_fixed_mean_file


SRR_TYPE_NUM = 6
SRR_TYPE_NAMES = ["Person", "Organization", "Location", "Time", "Number", "Miscellaneous"]


class HierarchyAwareEncoder(nn.Module):
    def __init__(self, hidden_size, use_cross_layer_attention=True,
                 use_sentence_position_encoding=True, max_sent_num=25,
                 paragraph_strategy="sentence_window", sentences_per_paragraph=4,
                 dropout=0.1):
        super().__init__()
        if paragraph_strategy not in {"single_document", "sentence_window"}:
            raise ValueError(f"Unknown paragraph strategy: {paragraph_strategy}")
        if sentences_per_paragraph <= 0:
            raise ValueError("sentences_per_paragraph must be positive.")
        if max_sent_num <= 0:
            raise ValueError("max_sent_num must be positive.")

        self.hidden_size = hidden_size
        self.use_cross_layer_attention = use_cross_layer_attention
        self.use_sentence_position_encoding = use_sentence_position_encoding
        self.max_sent_num = max_sent_num
        self.paragraph_strategy = paragraph_strategy
        self.sentences_per_paragraph = sentences_per_paragraph
        self.dropout = nn.Dropout(dropout)
        self.last_paragraph_attention = None

        if self.use_sentence_position_encoding:
            self.sentence_position_embedding = nn.Embedding(max_sent_num, hidden_size)

        self.doc_attn_w1 = nn.Linear(hidden_size, hidden_size)
        self.doc_attn_w2 = nn.Linear(hidden_size, 1)
        self.bottom_up_fusion = nn.Linear(hidden_size * 2, hidden_size)
        self.bottom_up_norm = nn.LayerNorm(hidden_size)
        self.token_delta = nn.Linear(hidden_size, hidden_size)
        self.token_norm = nn.LayerNorm(hidden_size)

        if self.use_cross_layer_attention:
            self.query = nn.Linear(hidden_size, hidden_size)
            self.key = nn.Linear(hidden_size, hidden_size)
            self.value = nn.Linear(hidden_size, hidden_size)
            self.context_out = nn.Linear(hidden_size, hidden_size)
            self.sentence_norm = nn.LayerNorm(hidden_size)

    def _pool_sentences(self, sequence_output, sent_pos, offset):
        batch_size, doc_len, hidden = sequence_output.size()
        max_sent_num = max([len(sents) for sents in sent_pos]) if sent_pos else 0
        sentence_repr = sequence_output.new_zeros(batch_size, max_sent_num, hidden)
        sentence_mask = torch.zeros(batch_size, max_sent_num, dtype=torch.bool, device=sequence_output.device)

        for batch_idx, doc_sent_pos in enumerate(sent_pos):
            for sent_idx, sent in enumerate(doc_sent_pos):
                start = min(sent[0] + offset, doc_len)
                end = min(sent[1] + offset, doc_len)
                if end > start:
                    sentence_repr[batch_idx, sent_idx] = sequence_output[batch_idx, start:end].mean(dim=0)
                    sentence_mask[batch_idx, sent_idx] = True

        return sentence_repr, sentence_mask

    def _add_sentence_position_encoding(self, sentence_repr, sentence_mask):
        if not self.use_sentence_position_encoding:
            return sentence_repr
        max_sent_num = sentence_repr.size(1)
        if max_sent_num > self.max_sent_num:
            raise ValueError(
                f"Batch contains {max_sent_num} sentences, exceeding max_sent_num={self.max_sent_num}."
            )
        position_ids = torch.arange(max_sent_num, device=sentence_repr.device)
        position_repr = self.sentence_position_embedding(position_ids).unsqueeze(0)
        sentence_repr = sentence_repr + position_repr
        return torch.where(sentence_mask.unsqueeze(-1), sentence_repr, sentence_repr.new_zeros(sentence_repr.size()))

    def _paragraph_id(self, sent_idx):
        if self.paragraph_strategy == "single_document":
            return 0
        return sent_idx // self.sentences_per_paragraph

    def _pool_paragraphs(self, sentence_repr, sentence_mask):
        batch_size, max_sent_num, hidden = sentence_repr.size()
        paragraph_ids = torch.full(
            (batch_size, max_sent_num),
            -1,
            dtype=torch.long,
            device=sentence_repr.device,
        )

        max_para_num = 0
        for batch_idx in range(batch_size):
            valid_sent_num = int(sentence_mask[batch_idx].sum().item())
            if valid_sent_num == 0:
                continue
            for sent_idx in range(valid_sent_num):
                paragraph_ids[batch_idx, sent_idx] = self._paragraph_id(sent_idx)
            max_para_num = max(max_para_num, int(paragraph_ids[batch_idx, :valid_sent_num].max().item()) + 1)

        paragraph_repr = sentence_repr.new_zeros(batch_size, max_para_num, hidden)
        paragraph_mask = torch.zeros(batch_size, max_para_num, dtype=torch.bool, device=sentence_repr.device)

        for batch_idx in range(batch_size):
            for para_idx in range(max_para_num):
                sent_mask = paragraph_ids[batch_idx].eq(para_idx) & sentence_mask[batch_idx]
                if sent_mask.any():
                    paragraph_repr[batch_idx, para_idx] = sentence_repr[batch_idx, sent_mask].mean(dim=0)
                    paragraph_mask[batch_idx, para_idx] = True

        return paragraph_repr, paragraph_mask, paragraph_ids

    def _pool_documents(self, paragraph_repr, paragraph_mask):
        batch_size, _, hidden = paragraph_repr.size()
        document_repr = paragraph_repr.new_zeros(batch_size, hidden)
        if paragraph_repr.size(1) == 0:
            self.last_paragraph_attention = paragraph_repr.new_zeros(batch_size, 0)
            return document_repr

        scores = self.doc_attn_w2(torch.tanh(self.doc_attn_w1(paragraph_repr))).squeeze(-1)
        scores = scores.masked_fill(~paragraph_mask, -1e4)
        alpha = torch.softmax(scores, dim=-1)
        alpha = torch.where(paragraph_mask, alpha, alpha.new_zeros(alpha.size()))
        alpha = alpha / (alpha.sum(dim=-1, keepdim=True) + 1e-30)
        document_repr = (alpha.unsqueeze(-1) * paragraph_repr).sum(dim=1)
        self.last_paragraph_attention = alpha.detach()
        return document_repr

    def _sentence_paragraph_context(self, sentence_repr, sentence_mask, paragraph_repr, paragraph_ids):
        batch_size, max_sent_num, hidden = sentence_repr.size()
        paragraph_context = sentence_repr.new_zeros(batch_size, max_sent_num, hidden)

        for batch_idx in range(batch_size):
            valid_sent = sentence_mask[batch_idx]
            if not valid_sent.any():
                continue
            valid_para_ids = paragraph_ids[batch_idx, valid_sent]
            paragraph_context[batch_idx, valid_sent] = paragraph_repr[batch_idx].index_select(0, valid_para_ids)

        return paragraph_context

    def _bottom_up_only(self, sentence_repr, sentence_mask, paragraph_repr, paragraph_ids):
        paragraph_context = self._sentence_paragraph_context(sentence_repr, sentence_mask, paragraph_repr, paragraph_ids)
        fused = self.bottom_up_fusion(torch.cat([sentence_repr, paragraph_context], dim=-1))
        enhanced = self.bottom_up_norm(sentence_repr + self.dropout(fused))
        return torch.where(sentence_mask.unsqueeze(-1), enhanced, sentence_repr.new_zeros(sentence_repr.size()))

    def _cross_layer_attention(self, sentence_repr, sentence_mask, paragraph_repr, paragraph_ids, document_repr):
        batch_size, max_sent_num, hidden = sentence_repr.size()
        paragraph_context = self._sentence_paragraph_context(sentence_repr, sentence_mask, paragraph_repr, paragraph_ids)

        document_context = document_repr.unsqueeze(1).expand(-1, max_sent_num, -1)
        q = self.query(sentence_repr)
        k = self.key(torch.stack([paragraph_context, document_context], dim=2))
        v = self.value(torch.stack([paragraph_context, document_context], dim=2))
        scores = (q.unsqueeze(2) * k).sum(dim=-1) / (hidden ** 0.5)
        scores = scores.masked_fill(~sentence_mask.unsqueeze(-1), -1e4)
        alpha = torch.softmax(scores, dim=-1)
        context = (alpha.unsqueeze(-1) * v).sum(dim=2)
        enhanced = self.sentence_norm(sentence_repr + self.dropout(self.context_out(context)))
        return torch.where(sentence_mask.unsqueeze(-1), enhanced, sentence_repr.new_zeros(sentence_repr.size()))

    def _inject_sentence_delta(self, sequence_output, sent_pos, offset, sentence_repr, enhanced_sentence_repr):
        batch_size, doc_len, hidden = sequence_output.size()
        token_delta = sequence_output.new_zeros(batch_size, doc_len, hidden)
        token_mask = torch.zeros(batch_size, doc_len, dtype=torch.bool, device=sequence_output.device)
        sentence_delta = self.token_delta(enhanced_sentence_repr - sentence_repr)

        for batch_idx, doc_sent_pos in enumerate(sent_pos):
            for sent_idx, sent in enumerate(doc_sent_pos):
                start = min(sent[0] + offset, doc_len)
                end = min(sent[1] + offset, doc_len)
                if end > start:
                    token_delta[batch_idx, start:end] = sentence_delta[batch_idx, sent_idx]
                    token_mask[batch_idx, start:end] = True

        injected = self.token_norm(sequence_output + self.dropout(token_delta))
        return torch.where(token_mask.unsqueeze(-1), injected, sequence_output)

    def forward(self, sequence_output, sent_pos, offset, fixed_document_repr=None):
        if sent_pos is None:
            raise ValueError("HAE requires sent_pos from DocRED preprocessing.")
        sentence_repr, sentence_mask = self._pool_sentences(sequence_output, sent_pos, offset)
        sentence_repr = self._add_sentence_position_encoding(sentence_repr, sentence_mask)
        paragraph_repr, paragraph_mask, paragraph_ids = self._pool_paragraphs(sentence_repr, sentence_mask)
        document_repr = self._pool_documents(paragraph_repr, paragraph_mask)
        if fixed_document_repr is not None:
            if fixed_document_repr.shape != (sequence_output.size(-1),):
                raise ValueError(
                    f"fixed_document_repr shape must be ({sequence_output.size(-1)},), got {tuple(fixed_document_repr.shape)}"
                )
            document_repr = fixed_document_repr.to(device=sequence_output.device, dtype=sequence_output.dtype).unsqueeze(0).expand(sequence_output.size(0), -1)

        if not self.use_cross_layer_attention:
            enhanced_sentence_repr = self._bottom_up_only(sentence_repr, sentence_mask, paragraph_repr, paragraph_ids)
            return self._inject_sentence_delta(
                sequence_output,
                sent_pos,
                offset,
                sentence_repr,
                enhanced_sentence_repr,
            )

        enhanced_sentence_repr = self._cross_layer_attention(
            sentence_repr,
            sentence_mask,
            paragraph_repr,
            paragraph_ids,
            document_repr,
        )
        return self._inject_sentence_delta(
            sequence_output,
            sent_pos,
            offset,
            sentence_repr,
            enhanced_sentence_repr,
        )


class StructuredRelationalReasoner(nn.Module):
    """Type-aware structural reasoning used by HDER.

    The module keeps the entity-type graph used by the relation branch while also
    exposing token-level type-aware attention, matching the formulation in the
    paper.  Token-level attention is blended conservatively with entity-level type
    pooling so the public implementation remains close to the experimentally used
    code path.
    """

    def __init__(self, hidden_size, steps=2, init="learned", adjacency_init=None,
                 dropout=0.1, token_attention_scale=0.05):
        super().__init__()
        if steps <= 0:
            raise ValueError("srr_steps must be positive.")
        if init not in {"learned", "cooccurrence"}:
            raise ValueError(f"Unknown SRR init: {init}")
        if token_attention_scale < 0:
            raise ValueError("token_attention_scale must be non-negative.")

        self.hidden_size = hidden_size
        self.steps = steps
        self.init = init
        self.type_num = SRR_TYPE_NUM
        self.token_attention_scale = float(token_attention_scale)
        self.dropout = nn.Dropout(dropout)

        self.type_fallback_embedding = nn.Embedding(self.type_num, hidden_size)
        self.type_query = nn.Parameter(torch.empty(self.type_num, hidden_size))
        nn.init.xavier_uniform_(self.type_query)

        # Entity-level pooling retained for numerical continuity with the original
        # experiment code. Token-level attention below adds the paper-aligned
        # q_e-to-h_ij interaction without replacing the stable entity aggregation.
        self.entity_pool_key = nn.Linear(hidden_size, hidden_size)
        self.entity_pool_value = nn.Linear(hidden_size, hidden_size)
        self.token_key = nn.Linear(hidden_size, hidden_size)
        self.token_value = nn.Linear(hidden_size, hidden_size)
        self.type_pool_norm = nn.LayerNorm(hidden_size)

        self.adjacency_logits = nn.Parameter(torch.zeros(self.type_num, self.type_num))
        if adjacency_init is not None:
            init_tensor = torch.as_tensor(adjacency_init, dtype=torch.float)
            if init_tensor.shape != (self.type_num, self.type_num):
                raise ValueError(f"SRR adjacency init must be 6x6, got {tuple(init_tensor.shape)}")
            init_tensor = init_tensor.clamp(1e-4, 1 - 1e-4)
            with torch.no_grad():
                self.adjacency_logits.copy_(torch.log(init_tensor / (1 - init_tensor)))

        self.message_query = nn.Linear(hidden_size, hidden_size)
        self.message_key = nn.Linear(hidden_size, hidden_size)
        self.message_value = nn.Linear(hidden_size, hidden_size)
        self.message_out = nn.Linear(hidden_size, hidden_size)
        self.message_norm = nn.LayerNorm(hidden_size)
        self.structural_project = nn.Linear(hidden_size * 3 + 1, hidden_size)
        self.fusion_norm = nn.LayerNorm(hidden_size)

        self.last_adjacency = None
        self.last_type_pooling_attention = None
        self.last_token_type_attention = None
        self.last_type_attention = None
        self.last_type_nodes = None

    def _validate_entity_types(self, entity_reprs, entity_types):
        if len(entity_reprs) != len(entity_types):
            raise ValueError(
                f"entity_repr/entity_type batch size mismatch: {len(entity_reprs)} vs {len(entity_types)}"
            )
        for doc_idx, (doc_repr, doc_types) in enumerate(zip(entity_reprs, entity_types)):
            if doc_repr.size(0) != len(doc_types):
                raise ValueError(
                    f"entity_type length mismatch at doc {doc_idx}: "
                    f"{len(doc_types)} types for {doc_repr.size(0)} entities"
                )
            for type_id in doc_types:
                if not isinstance(type_id, int) or type_id < 0 or type_id >= self.type_num:
                    raise ValueError(f"Invalid SRR entity type id at doc {doc_idx}: {type_id}")

    def _build_entity_type_nodes(self, entity_repr, entity_type):
        device = entity_repr.device
        type_ids = torch.tensor(entity_type, dtype=torch.long, device=device)
        fallback_ids = torch.arange(self.type_num, device=device)
        fallback_nodes = self.type_fallback_embedding(fallback_ids)
        pooled_keys = self.entity_pool_key(entity_repr)
        pooled_values = self.entity_pool_value(entity_repr)
        pooling_attention = entity_repr.new_zeros(self.type_num, entity_repr.size(0))
        type_nodes = []

        for type_id in range(self.type_num):
            mask = type_ids.eq(type_id)
            if mask.any():
                score = (pooled_keys[mask] * self.type_query[type_id]).sum(dim=-1) / (self.hidden_size ** 0.5)
                alpha = torch.softmax(score, dim=0)
                pooling_attention[type_id, mask] = alpha
                type_nodes.append((alpha.unsqueeze(-1) * pooled_values[mask]).sum(dim=0))
            else:
                type_nodes.append(fallback_nodes[type_id])

        return torch.stack(type_nodes, dim=0), pooling_attention

    def _build_token_type_nodes(self, token_repr, token_mask, adjacency):
        if token_mask is None:
            token_mask = torch.ones(token_repr.size(0), dtype=torch.bool, device=token_repr.device)
        else:
            token_mask = token_mask.to(device=token_repr.device, dtype=torch.bool)

        if not token_mask.any():
            fallback_ids = torch.arange(self.type_num, device=token_repr.device)
            fallback = self.type_fallback_embedding(fallback_ids)
            empty_attn = token_repr.new_zeros(self.type_num, token_repr.size(0))
            return fallback, empty_attn

        keys = self.token_key(token_repr)
        values = self.token_value(token_repr)
        base_score = torch.matmul(self.type_query, keys.transpose(0, 1)) / (self.hidden_size ** 0.5)

        # Neighbour-type contribution mirrors the paper's adjacency-conditioned
        # type-aware attention term. Row-normalization keeps the scale stable.
        adj_norm = adjacency / (adjacency.sum(dim=-1, keepdim=True) + 1e-6)
        neighbour_query = torch.matmul(adj_norm, self.type_query)
        neighbour_score = torch.matmul(neighbour_query, keys.transpose(0, 1)) / (self.hidden_size ** 0.5)
        beta = base_score + neighbour_score
        beta = beta.masked_fill(~token_mask.unsqueeze(0), -1e4)
        alpha = torch.softmax(beta, dim=-1)
        alpha = torch.where(token_mask.unsqueeze(0), alpha, alpha.new_zeros(alpha.size()))
        alpha = alpha / (alpha.sum(dim=-1, keepdim=True) + 1e-30)
        token_nodes = torch.matmul(alpha, values)
        return token_nodes, alpha

    def _message_pass(self, type_nodes, adjacency):
        attention_steps = []
        for _ in range(self.steps):
            query = self.message_query(type_nodes)
            key = self.message_key(type_nodes)
            value = self.message_value(type_nodes)
            semantic_score = torch.matmul(query, key.transpose(0, 1)) / (self.hidden_size ** 0.5)
            prior_score = torch.log(adjacency + 1e-6)
            beta = semantic_score + prior_score
            alpha = torch.softmax(beta, dim=-1)
            if not torch.isfinite(alpha).all():
                raise ValueError("SRR type attention produced non-finite values.")
            message = torch.matmul(alpha, value)
            type_nodes = self.message_norm(type_nodes + self.dropout(self.message_out(message)))
            if not torch.isfinite(type_nodes).all():
                raise ValueError("SRR message passing produced non-finite type nodes.")
            attention_steps.append(alpha.detach())
        self.last_type_attention = torch.stack(attention_steps, dim=0)
        return type_nodes

    def forward(self, pair_repr, entity_reprs, entity_types, pair_indices, rels_per_batch,
                token_reprs=None, token_masks=None):
        self._validate_entity_types(entity_reprs, entity_types)
        if sum(rels_per_batch) != pair_repr.size(0):
            raise ValueError(
                f"sum(rels_per_batch)={sum(rels_per_batch)} does not match pair_repr rows={pair_repr.size(0)}."
            )
        if token_reprs is None:
            raise ValueError("SRR token-level type attention requires token_reprs.")

        adjacency = torch.sigmoid(self.adjacency_logits)
        self.last_adjacency = adjacency.detach()
        if not torch.isfinite(adjacency).all():
            raise ValueError("SRR adjacency contains non-finite values.")

        enhanced_pairs = []
        all_entity_pool_attn = []
        all_token_type_attn = []
        all_type_nodes = []
        offset = 0

        for doc_idx, (doc_entity_repr, doc_entity_type, doc_pairs) in enumerate(
                zip(entity_reprs, entity_types, pair_indices)):
            rel_count = rels_per_batch[doc_idx]
            if len(doc_pairs) != rel_count:
                raise ValueError(
                    f"len(doc_pairs)={len(doc_pairs)} does not match rel_count={rel_count} at doc {doc_idx}."
                )
            entity_count = doc_entity_repr.size(0)
            for h, t in doc_pairs:
                if h < 0 or h >= entity_count or t < 0 or t >= entity_count:
                    raise ValueError(
                        f"SRR pair index out of range at doc {doc_idx}: ({h}, {t}) for {entity_count} entities."
                    )

            doc_pair_repr = pair_repr[offset:offset + rel_count]
            entity_nodes, entity_attn = self._build_entity_type_nodes(doc_entity_repr, doc_entity_type)
            doc_token_mask = None if token_masks is None else token_masks[doc_idx]
            token_nodes, token_attn = self._build_token_type_nodes(token_reprs[doc_idx], doc_token_mask, adjacency)

            # A small token-level residual preserves the behaviour of the existing
            # relation path while adding the paper-aligned token/type interaction.
            type_nodes = self.type_pool_norm(entity_nodes + self.token_attention_scale * token_nodes)
            type_nodes = self._message_pass(type_nodes, adjacency)

            head_types = torch.tensor(
                [doc_entity_type[h] for h, _ in doc_pairs],
                dtype=torch.long,
                device=pair_repr.device,
            )
            tail_types = torch.tensor(
                [doc_entity_type[t] for _, t in doc_pairs],
                dtype=torch.long,
                device=pair_repr.device,
            )
            head_type_nodes = type_nodes.index_select(0, head_types)
            tail_type_nodes = type_nodes.index_select(0, tail_types)
            compatibility = adjacency[head_types, tail_types].unsqueeze(-1)
            structural_input = torch.cat(
                [doc_pair_repr, head_type_nodes, tail_type_nodes, compatibility],
                dim=-1,
            )
            structural_repr = self.structural_project(structural_input)
            doc_enhanced = self.fusion_norm(doc_pair_repr + self.dropout(structural_repr))
            if not torch.isfinite(doc_enhanced).all():
                raise ValueError("SRR enhanced pair representation contains non-finite values.")
            enhanced_pairs.append(doc_enhanced)
            all_entity_pool_attn.append(entity_attn.detach())
            all_token_type_attn.append(token_attn.detach())
            all_type_nodes.append(type_nodes)
            offset += rel_count

        if offset != pair_repr.size(0):
            raise ValueError(f"SRR consumed {offset} pairs, but pair_repr has {pair_repr.size(0)} rows.")
        output = torch.cat(enhanced_pairs, dim=0)
        if output.shape != pair_repr.shape:
            raise ValueError(f"SRR final output shape mismatch: {tuple(output.shape)} vs {tuple(pair_repr.shape)}")

        self.last_type_pooling_attention = all_entity_pool_attn
        self.last_token_type_attention = all_token_type_attn
        self.last_type_nodes = all_type_nodes
        return output


class TypedSpanBoundaryHead(nn.Module):
    """Typed start/end span decoder used as a lightweight auxiliary head.

    The head follows the paper's h_ij ⊕ lambda_e r_e fusion. By default its
    loss is detached from the backbone/SRR path so adding the decoder does not
    materially perturb relation-extraction scores in the public training recipe.
    """

    def __init__(self, hidden_size, type_num=SRR_TYPE_NUM):
        super().__init__()
        self.hidden_size = hidden_size
        self.type_num = type_num
        self.type_scale = nn.Parameter(torch.ones(type_num))
        self.start_weight = nn.Parameter(torch.empty(type_num, hidden_size * 2))
        self.end_weight = nn.Parameter(torch.empty(type_num, hidden_size * 2))
        self.start_bias = nn.Parameter(torch.zeros(type_num))
        self.end_bias = nn.Parameter(torch.zeros(type_num))
        nn.init.xavier_uniform_(self.start_weight)
        nn.init.xavier_uniform_(self.end_weight)

    def forward(self, token_repr, type_nodes):
        # token_repr: [L, D], type_nodes: [T, D]
        length = token_repr.size(0)
        token_expand = token_repr.unsqueeze(1).expand(length, self.type_num, self.hidden_size)
        structural = (self.type_scale.unsqueeze(-1) * type_nodes).unsqueeze(0).expand(length, -1, -1)
        fused = torch.cat([token_expand, structural], dim=-1)
        start_logits = (fused * self.start_weight.unsqueeze(0)).sum(dim=-1) + self.start_bias
        end_logits = (fused * self.end_weight.unsqueeze(0)).sum(dim=-1) + self.end_bias
        return start_logits, end_logits


class HDERModel(nn.Module):

    def __init__(self, config, model, tokenizer,
                emb_size=768, block_size=64, num_labels=-1,
                max_sent_num=25, evi_thresh=0.2,
                use_hae=False, use_cross_layer_attention=False,
                global_context_mode="dynamic", use_srr=False,
                srr_steps=2, srr_init="learned", srr_adjacency_init=None,
                global_mean_path="", max_seq_length=1024,
                use_sentence_position_encoding=True,
                paragraph_strategy="sentence_window", sentences_per_paragraph=4,
                use_span_boundary_head=True, span_detach_backbone=True,
                srr_token_attention_scale=0.05):
        '''
        Initialize the model.
        :model: Pretrained langage model encoder;
        :tokenizer: Tokenzier corresponding to the pretrained language model encoder;
        :emb_size: Dimension of embeddings for subject/object (head/tail) representations;
        :block_size: Number of blocks for grouped bilinear classification;
        :num_labels: Maximum number of relation labels for each entity pair;
        :max_sent_num: Maximum number of sentences for each document;
        :evi_thresh: Threshold for selecting evidence sentences.
        '''
        
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_size = config.hidden_size

        self.loss_fnt = ATLoss()
        self.loss_fnt_evi = nn.KLDivLoss(reduction="batchmean")

        self.head_extractor = nn.Linear(self.hidden_size * 2, emb_size)
        self.tail_extractor = nn.Linear(self.hidden_size * 2, emb_size) 
        self.head_extractor2 = nn.Linear(emb_size + self.hidden_size * 2, emb_size)
        self.tail_extractor2 = nn.Linear(emb_size + self.hidden_size * 2, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)
        # self.bilinear = nn.Linear(self.hidden_size * , config.num_labels)
        self.bilinear2 = nn.Linear(emb_size * block_size, self.hidden_size)
        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels
        self.total_labels = config.num_labels
        self.max_sent_num = max_sent_num
        self.evi_thresh = evi_thresh
        self.use_hae = use_hae
        self.use_cross_layer_attention = use_cross_layer_attention
        self.global_context_mode = global_context_mode
        self.use_srr = use_srr
        self.srr_steps = srr_steps
        self.srr_init = srr_init
        self.global_mean_path = global_mean_path
        self.max_seq_length = max_seq_length
        self.fixed_global_mean_metadata = None
        self.use_sentence_position_encoding = use_sentence_position_encoding
        self.paragraph_strategy = paragraph_strategy
        self.sentences_per_paragraph = sentences_per_paragraph
        self.use_span_boundary_head = use_span_boundary_head
        self.span_detach_backbone = span_detach_backbone
        self.srr_token_attention_scale = srr_token_attention_scale
        self.hae = None
        self.srr = None
        self.span_boundary_head = None
        if self.global_context_mode == "fixed_mean":
            mean_tensor, metadata = self._load_fixed_global_mean(global_mean_path, max_seq_length)
            self.register_buffer("fixed_global_mean", mean_tensor)
            self.fixed_global_mean_metadata = metadata
        if self.use_cross_layer_attention and not self.use_hae:
            raise ValueError("use_cross_layer_attention requires use_hae=true.")
        if self.use_hae:
            self.hae = HierarchyAwareEncoder(
                self.hidden_size,
                use_cross_layer_attention=self.use_cross_layer_attention,
                use_sentence_position_encoding=self.use_sentence_position_encoding,
                max_sent_num=self.max_sent_num,
                paragraph_strategy=self.paragraph_strategy,
                sentences_per_paragraph=self.sentences_per_paragraph,
            )
        if self.use_srr:
            self.srr = StructuredRelationalReasoner(
                self.hidden_size,
                steps=self.srr_steps,
                init=self.srr_init,
                adjacency_init=srr_adjacency_init,
                token_attention_scale=self.srr_token_attention_scale,
            )
        if self.use_span_boundary_head and self.use_srr:
            self.span_boundary_head = TypedSpanBoundaryHead(self.hidden_size)

    def _load_fixed_global_mean(self, global_mean_path, max_seq_length):
        if not global_mean_path:
            raise ValueError("--global_mean_path is required when --global_context_mode fixed_mean.")
        current_model_name = getattr(self.config, "_name_or_path", None) or getattr(self.config, "name_or_path", None)
        result = validate_fixed_mean_file(
            global_mean_path,
            hidden_size=self.hidden_size,
            max_seq_length=max_seq_length,
            current_config=self.config,
            current_model_name=current_model_name,
        )
        mean = result["mean"]
        metadata = result["metadata"]
        print(f"FIXED_MEAN_LOADED: {global_mean_path}")
        print(f"FIXED_MEAN_METADATA: {metadata}")
        return mean, metadata
    def encode(self, input_ids, attention_mask):
        
        '''
        Get the embedding of each token. For long document that has more than 512 tokens, split it into two overlapping chunks.
        Inputs:
            :input_ids: (batch_size, doc_len)
            :attention_mask: (batch_size, doc_len)
        Outputs:
            :sequence_output: (batch_size, doc_len, hidden_dim)
            :attention: (batch_size, num_attn_heads, doc_len, doc_len)
        '''
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        # process long documents.
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts, offset):

        '''
        Get head, tail, context embeddings from token embeddings.
        Inputs:
            :sequence_output: (batch_size, doc_len, hidden_dim)
            :attention: (batch_size, num_attn_heads, doc_len, doc_len)
            :entity_pos: list of list. Outer length = batch size, inner length = number of entities each batch.
            :hts: list of list. Outer length = batch size, inner length = number of combination of entity pairs each batch.
            :offset: 1 for bert and roberta. Offset caused by [CLS] token.
        Outputs:
            :hss: (num_ent_pairs_all_batches, emb_size)
            :tss: (num_ent_pairs_all_batches, emb_size)
            :rss: (num_ent_pairs_all_batches, emb_size)
            :ht_atts: (num_ent_pairs_all_batches, doc_len)
            :rels_per_batch: list of length = batch size. Each entry represents the number of entity pairs of the batch.
        '''
        
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        ht_atts = []
        entity_embs_per_doc = []

        for i in range(len(entity_pos)): # for each batch
            entity_embs, entity_atts = [], []
            
            # obtain entity embedding from mention embeddings.
            for eid, e in enumerate(entity_pos[i]): # for each entity
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for mid, (start, end) in enumerate(e): # for every mention
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset]) 
                            e_att.append(attention[i, :, start + offset]) 

                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0) 
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)

                entity_embs.append(e_emb) 
                entity_atts.append(e_att) 
                
            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]
            entity_embs_per_doc.append(entity_embs)

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)

            # obtain subject/object (head/tail) embeddings from entity embeddings.
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])
                
            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])

            ht_att = (h_att * t_att).mean(1) # average over all heads  （num_ent_pairs,doc_len）       
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-30) 
            ht_atts.append(ht_att) 
            
            # obtain local context embeddings.
            rs = contract("ld,rl->rd", sequence_output[i], ht_att) 

            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
        
        rels_per_batch = [len(b) for b in hss]
        hss = torch.cat(hss, dim=0) # (num_ent_pairs_all_batches, emb_size)
        tss = torch.cat(tss, dim=0) # (num_ent_pairs_all_batches, emb_size)
        rss = torch.cat(rss, dim=0) # (num_ent_pairs_all_batches, emb_size)
        ht_atts = torch.cat(ht_atts, dim=0) # (num_ent_pairs_all_batches, max_doc_len)

        return hss, rss, tss, ht_atts, rels_per_batch, entity_embs_per_doc

    def forward_rel(self, hs, ts, rs, batch_rel, entity_reprs=None, entity_type=None, pair_indices=None,
                    token_reprs=None, token_masks=None):
        '''
        Forward computation for RE.
        Inputs:
            :hs: (num_ent_pairs_all_batches, emb_size)
            :ts: (num_ent_pairs_all_batches, emb_size)
            :rs: (num_ent_pairs_all_batches, emb_size)
        Outputs:
            :logits: (num_ent_pairs_all_batches, num_rel_labels)
        '''
        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=-1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=-1)))
        # split into several groups.
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)

        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        hts = self.bilinear2(bl)
        if self.srr is not None:
            if entity_reprs is None or entity_type is None or pair_indices is None:
                raise ValueError("SRR requires entity_reprs, entity_type, and pair_indices.")
            hts = self.srr(
                hts, entity_reprs, entity_type, pair_indices, batch_rel,
                token_reprs=token_reprs, token_masks=token_masks,
            )

        hs = torch.tanh(self.head_extractor2(torch.cat([hs, rs, hts], dim=-1)))
        ts = torch.tanh(self.tail_extractor2(torch.cat([ts, rs, hts], dim=-1)))
         # split into several groups.
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)
        
        return logits


    def forward_evi(self, doc_attn, sent_pos, batch_rel, offset):
        '''
        Forward computation for ER.
        Inputs:
            :doc_attn: (num_ent_pairs_all_batches, doc_len), attention weight of each token for computing localized context pooling.
            :sent_pos: list of list. The outer length = batch size. The inner list contains (start, end) position of each sentence in each batch.
            :batch_rel: list of length = batch size. Each entry represents the number of entity pairs of the batch.
            :offset: 1 for bert and roberta. Offset caused by [CLS] token.
        Outputs:
            :s_attn:  (num_ent_pairs_all_batches, max_sent_all_batch), sentence-level evidence distribution of each entity pair.
        '''
        
        max_sent_num = max([len(sent) for sent in sent_pos])
        rel_sent_attn = []
        for i in range(len(sent_pos)): # for each batch
            # the relation ids corresponds to document in batch i is [sum(batch_rel[:i]), sum(batch_rel[:i+1]))
            curr_attn = doc_attn[sum(batch_rel[:i]):sum(batch_rel[:i+1])]
            doc_len = curr_attn.size(-1)
            curr_sent_pos = []
            for s in sent_pos[i]:
                start = min(s[0] + offset, doc_len)
                end = min(s[1] + offset, doc_len)
                if end > start:
                    curr_sent_pos.append(torch.arange(start, end).to(curr_attn.device))
                else:
                    curr_sent_pos.append(None)

            curr_attn_per_sent = []
            empty_sent = torch.zeros(curr_attn.size(0), 1, device=curr_attn.device, dtype=curr_attn.dtype)
            for sent in curr_sent_pos:
                if sent is None:
                    curr_attn_per_sent.append(empty_sent)
                else:
                    curr_attn_per_sent.append(curr_attn.index_select(-1, sent))
            curr_attn_per_sent += [empty_sent] * (max_sent_num - len(curr_attn_per_sent))
            sum_attn = torch.stack([attn.sum(dim=-1) for attn in curr_attn_per_sent], dim=-1) # sum across those attentions
            rel_sent_attn.append(sum_attn)

        s_attn = torch.cat(rel_sent_attn, dim=0)
        return s_attn


    def _build_span_boundary_targets(self, entity_pos, entity_type, seq_len, offset, device):
        start_targets = torch.zeros(len(entity_pos), seq_len, SRR_TYPE_NUM, device=device)
        end_targets = torch.zeros_like(start_targets)
        for doc_idx, (doc_entities, doc_types) in enumerate(zip(entity_pos, entity_type)):
            if len(doc_entities) != len(doc_types):
                raise ValueError(
                    f"Span target entity/type mismatch at doc {doc_idx}: "
                    f"{len(doc_entities)} entities vs {len(doc_types)} types."
                )
            for ent_idx, mentions in enumerate(doc_entities):
                type_id = doc_types[ent_idx]
                if type_id < 0 or type_id >= SRR_TYPE_NUM:
                    continue
                for start, end in mentions:
                    start_idx = start + offset
                    end_idx = end - 1 + offset
                    if 0 <= start_idx < seq_len:
                        start_targets[doc_idx, start_idx, type_id] = 1.0
                    if 0 <= end_idx < seq_len:
                        end_targets[doc_idx, end_idx, type_id] = 1.0
        return start_targets, end_targets

    def _span_boundary_forward(self, sequence_output, attention_mask):
        if self.span_boundary_head is None or self.srr is None or self.srr.last_type_nodes is None:
            return None, None
        start_logits, end_logits = [], []
        for doc_idx, type_nodes in enumerate(self.srr.last_type_nodes):
            token_repr = sequence_output[doc_idx]
            if self.span_detach_backbone:
                token_repr = token_repr.detach()
                type_nodes = type_nodes.detach()
            s_logit, e_logit = self.span_boundary_head(token_repr, type_nodes)
            if attention_mask is not None:
                valid = attention_mask[doc_idx].to(dtype=torch.bool, device=s_logit.device)
                s_logit = s_logit.masked_fill(~valid.unsqueeze(-1), -20.0)
                e_logit = e_logit.masked_fill(~valid.unsqueeze(-1), -20.0)
            start_logits.append(s_logit)
            end_logits.append(e_logit)
        return torch.stack(start_logits, dim=0), torch.stack(end_logits, dim=0)

    def _span_boundary_loss(self, start_logits, end_logits, entity_pos, entity_type,
                            attention_mask, offset):
        if start_logits is None or end_logits is None:
            return None
        seq_len = start_logits.size(1)
        start_targets, end_targets = self._build_span_boundary_targets(
            entity_pos, entity_type, seq_len, offset, start_logits.device
        )
        if attention_mask is None:
            valid = torch.ones(start_logits.size(0), seq_len, dtype=torch.bool, device=start_logits.device)
        else:
            valid = attention_mask[:, :seq_len].to(device=start_logits.device, dtype=torch.bool)
        valid = valid.unsqueeze(-1).expand_as(start_logits)
        start_loss = F.binary_cross_entropy_with_logits(
            start_logits[valid], start_targets[valid], reduction="mean"
        )
        end_loss = F.binary_cross_entropy_with_logits(
            end_logits[valid], end_targets[valid], reduction="mean"
        )
        return 0.5 * (start_loss + end_loss)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None, # relation labels
                entity_pos=None,
                entity_type=None,
                hts=None, # entity pairs
                sent_pos=None, 
                sent_labels=None, # evidence labels (0/1)
                teacher_attns=None, # evidence distribution from teacher model
                tag="train",
                ):

        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        output = {}
        sequence_output, attention = self.encode(input_ids, attention_mask)
        if self.hae is not None:
            fixed_document_repr = getattr(self, "fixed_global_mean", None)
            sequence_output = self.hae(sequence_output, sent_pos, offset, fixed_document_repr=fixed_document_repr)
        hs, rs, ts, doc_attn, batch_rel, entity_reprs = self.get_hrt(sequence_output, attention, entity_pos, hts, offset)
        logits = self.forward_rel(
            hs,
            ts,
            rs,
            batch_rel,
            entity_reprs=entity_reprs,
            entity_type=entity_type,
            pair_indices=hts,
            token_reprs=sequence_output,
            token_masks=attention_mask,
        )
        span_start_logits, span_end_logits = self._span_boundary_forward(sequence_output, attention_mask)
        output["rel_pred"] = self.loss_fnt.get_label(logits, num_labels=self.num_labels)

        if sent_labels is not None: # human-annotated evidence available

            s_attn = self.forward_evi(doc_attn, sent_pos, batch_rel, offset) 
            output["evi_pred"] = F.pad(s_attn > self.evi_thresh, (0, self.max_sent_num - s_attn.shape[-1])) 

        if tag in ["test", "dev"]: # testing
            scores_topk = self.loss_fnt.get_score(logits, self.num_labels) 
            output["scores"] = scores_topk[0]
            output["topks"] = scores_topk[1]
        
        if tag == "infer": # teacher model inference
            output["attns"] = doc_attn.split(batch_rel)

        else: # training
            # relation extraction loss
            loss = self.loss_fnt(logits.float(), labels.float())
            output["loss"] = {"rel_loss": loss.to(sequence_output)}
            if (self.use_span_boundary_head and span_start_logits is not None
                    and entity_pos is not None and entity_type is not None):
                span_loss = self._span_boundary_loss(
                    span_start_logits, span_end_logits, entity_pos, entity_type, attention_mask, offset
                )
                if span_loss is not None:
                    output["loss"]["span_loss"] = span_loss.to(sequence_output)
            if sent_labels is not None: # supervised training with human evidence

                idx_used = torch.nonzero(labels[:,1:].sum(dim=-1)).view(-1)
                # evidence retrieval loss (kldiv loss)
                s_attn = s_attn[idx_used]
                sent_labels = sent_labels[idx_used]
                norm_s_labels = sent_labels/(sent_labels.sum(dim=-1, keepdim=True) + 1e-30)
                norm_s_labels[norm_s_labels == 0] = 1e-30
                s_attn[s_attn == 0] = 1e-30
                evi_loss = self.loss_fnt_evi(s_attn.log(), norm_s_labels)
                output["loss"]["evi_loss"] = evi_loss.to(sequence_output)
            
            elif teacher_attns is not None: # self training with teacher attention
                
                doc_attn[doc_attn == 0] = 1e-30
                teacher_attns[teacher_attns == 0] = 1e-30
                attn_loss = self.loss_fnt_evi(doc_attn.log(), teacher_attns)
                output["loss"]["attn_loss"] = attn_loss.to(sequence_output)
        
        return output
