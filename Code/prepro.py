from tqdm import tqdm
try:
    import ujson as json
except ImportError:
    import json
import numpy as np
import pickle
import os
# Resolve rel2id.json relative to the HDER source directory.
import os as _os
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
try:
    docred_rel2id = json.load(open(_os.path.join(_SCRIPT_DIR, 'dataset/meta/rel2id.json'), 'r'))
except FileNotFoundError:
    # will fail later with a clear message when actually used
    docred_rel2id = {}
docred_ent2id = {'NA': 0, 'ORG': 1, 'LOC': 2, 'NUM': 3, 'TIME': 4, 'MISC': 5, 'PER': 6}
SRR_ENTITY_TYPE_TO_ID = {
    "PER": 0,
    "ORG": 1,
    "LOC": 2,
    "TIME": 3,
    "NUM": 4,
    "MISC": 5,
}
SRR_ENTITY_TYPE_NAMES = ["Person", "Organization", "Location", "Time", "Number", "Miscellaneous"]


def resolve_srr_entity_type(entity, title="", entity_idx=-1, conflict_policy="majority_first"):
    mention_types = [mention["type"] for mention in entity]
    mention_names = [mention.get("name", "") for mention in entity]
    for entity_type in mention_types:
        if entity_type not in SRR_ENTITY_TYPE_TO_ID:
            raise ValueError(
                f"Unknown DocRED entity type '{entity_type}' in document '{title}', "
                f"entity_idx={entity_idx}."
            )

    conflict = len(set(mention_types)) != 1
    if not conflict:
        return SRR_ENTITY_TYPE_TO_ID[mention_types[0]], None

    if conflict_policy == "strict":
        raise ValueError(
            f"Inconsistent entity mention types in document '{title}', "
            f"entity_idx={entity_idx}: {mention_types}"
        )
    if conflict_policy != "majority_first":
        raise ValueError(f"Unknown entity_type_conflict_policy: {conflict_policy}")

    counts = {}
    for entity_type in mention_types:
        counts[entity_type] = counts.get(entity_type, 0) + 1
    max_count = max(counts.values())
    tied_types = {entity_type for entity_type, count in counts.items() if count == max_count}
    chosen_type = next(entity_type for entity_type in mention_types if entity_type in tied_types)
    reason = "majority" if len(tied_types) == 1 else "tie_first_mention"
    record = {
        "title": title,
        "entity_idx": entity_idx,
        "mention_names": mention_names,
        "mention_types": mention_types,
        "chosen_type": chosen_type,
        "resolution_reason": reason,
    }
    return SRR_ENTITY_TYPE_TO_ID[chosen_type], record


def get_srr_entity_types(entities, title="", conflict_policy="majority_first", conflict_records=None):
    entity_types = []
    for entity_idx, entity in enumerate(entities):
        entity_type, record = resolve_srr_entity_type(entity, title, entity_idx, conflict_policy)
        if record is not None and conflict_records is not None:
            conflict_records.append(record)
        entity_types.append(entity_type)
    return entity_types


def print_entity_type_conflict_summary(file_in, policy, total_entities, conflict_records, conflict_log_dir=""):
    basename = os.path.basename(file_in)
    print("ENTITY_TYPE_CONFLICT_SUMMARY:")
    print(f"file={basename}")
    print(f"policy={policy}")
    print(f"total_entities={total_entities}")
    print(f"conflict_entities={len(conflict_records)}")
    for record in conflict_records[:5]:
        print(
            "conflict_sample="
            + json.dumps(
                {
                    "title": record["title"],
                    "entity_idx": record["entity_idx"],
                    "mention_names": record["mention_names"],
                    "mention_types": record["mention_types"],
                    "chosen_type": record["chosen_type"],
                    "resolution_reason": record["resolution_reason"],
                },
                ensure_ascii=False,
            )
        )
    if conflict_log_dir and conflict_records:
        os.makedirs(conflict_log_dir, exist_ok=True)
        stem = os.path.splitext(basename)[0]
        out_path = os.path.join(conflict_log_dir, f"entity_type_conflicts_{stem}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(conflict_records, fh, ensure_ascii=False, indent=2)
        print(f"ENTITY_TYPE_CONFLICT_LOG_SAVED: {out_path}")

def build_inputs_with_special_tokens_compat(tokenizer, input_ids):
    if hasattr(tokenizer, "build_inputs_with_special_tokens"):
        return tokenizer.build_inputs_with_special_tokens(input_ids)
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    if cls_id is None or sep_id is None:
        raise AttributeError("Tokenizer must provide cls_token_id and sep_token_id.")
    return [cls_id] + list(input_ids) + [sep_id]

def add_entity_markers(sample, tokenizer, entity_start, entity_end):
    ''' add entity marker (*) at the end and beginning of entities. '''

    sents = []
    sent_map = []
    sent_pos = []

    sent_start = 0
    for i_s, sent in enumerate(sample['sents']):
    # add * marks to the beginning and end of entities
        new_map = {}
        
        for i_t, token in enumerate(sent):
            tokens_wordpiece = tokenizer.tokenize(token)
            if (i_s, i_t) in entity_start:
                tokens_wordpiece = ["*"] + tokens_wordpiece
            if (i_s, i_t) in entity_end:
                tokens_wordpiece = tokens_wordpiece + ["*"]
            new_map[i_t] = len(sents)
            sents.extend(tokens_wordpiece)
        
        sent_end = len(sents)
        # [sent_start, sent_end)
        sent_pos.append((sent_start, sent_end,))
        sent_start = sent_end
        
        # update the start/end position of each token.
        new_map[i_t + 1] = len(sents)
        sent_map.append(new_map)

    return sents, sent_map, sent_pos

def get_pseudo_features(raw_feature: dict, pred_rels: list, entities: list, sent_map: dict, offset: int, tokenizer = None): 

    ''' Construct pseudo documents from predictions.'''
    
    pos_samples = 0
    neg_samples = 0
    
    sent_grps = []
    pseudo_features = []
    raw_entity_types = raw_feature.get("entity_type")

    for pred_rel in pred_rels:
        curr_sents = pred_rel["evidence"] #evidence sentence
        if len(curr_sents) == 0:
            continue

        # check if head/tail entity presents in evidence. if not, append sentence containing the first mention of head/tail into curr_sents
        head_sents = sorted([m["sent_id"] for m in entities[pred_rel["h_idx"]]]) 
        tail_sents = sorted([m["sent_id"] for m in entities[pred_rel["t_idx"]]]) #same

        if len(set(head_sents) & set(curr_sents)) == 0: 
            curr_sents.append(head_sents[0]) 
        if len(set(tail_sents) & set(curr_sents)) == 0:  
            curr_sents.append(tail_sents[0])

        curr_sents = sorted(set(curr_sents)) 
        if curr_sents in sent_grps: # skip if such sentence group has already been created
            continue
        sent_grps.append(curr_sents)

        # new sentence masks and input ids
        old_sent_pos = [raw_feature["sent_pos"][i] for i in curr_sents] 
        new_input_ids_each = [raw_feature["input_ids"][s[0] + offset:s[1] + offset] for s in old_sent_pos] 
        new_input_ids = sum(new_input_ids_each, [])
        new_input_ids = build_inputs_with_special_tokens_compat(tokenizer, new_input_ids)
 
        new_sent_pos = []

        prev_len = 0
        for sent in old_sent_pos: 
            curr_sent_pos =  (prev_len, prev_len + sent[1] - sent[0])
            new_sent_pos.append(curr_sent_pos)
            prev_len += sent[1] - sent[0]

        # iterate through all entities, keep only entities with mention in curr_sents.
        
        # obtain entity positions w.r.t whole document
        curr_entities = []  
        ent_new2old = {} # head/tail of a relation should be selected
        new_entity_pos = []
        new_entity_type = []

        for i, entity in enumerate(entities):
            curr = []
            curr_pos = []
            for mention in entity:
                if mention["sent_id"] in curr_sents:
                    curr.append(mention)
                    prev_len = new_sent_pos[curr_sents.index(mention["sent_id"])][0] 
                    pos = [sent_map[mention["sent_id"]][pos] - sent_map[mention["sent_id"]][0] + prev_len for pos in mention['pos']]
                    curr_pos.append(pos)

            if curr != []:
                curr_entities.append(curr)
                new_entity_pos.append(curr_pos)
                if raw_entity_types is not None:
                    new_entity_type.append(raw_entity_types[i])
                ent_new2old[len(ent_new2old)] = i # update dictionary
        
        # iterate through all entities to obtain all entity pairs
        new_hts = []
        new_labels = []
        for h in range(len(curr_entities)):
            for t in range(len(curr_entities)):
                if h != t:
                    new_hts.append([h, t])
                    old_h, old_t = ent_new2old[h], ent_new2old[t]
                    curr_label = raw_feature["labels"][raw_feature["hts"].index([old_h, old_t])]
                    new_labels.append(curr_label)

                    neg_samples += curr_label[0]
                    pos_samples += 1 - curr_label[0]
        pseudo_feature = {'input_ids': new_input_ids,
                    'entity_pos': new_entity_pos,
                    'labels': new_labels,
                    'hts': new_hts,
                    'entity_type': new_entity_type if raw_entity_types is not None else None,
                    'sent_pos': new_sent_pos,
                    'sent_labels': None,
                    'title': raw_feature['title'],
                    'entity_map': ent_new2old
                    }
        pseudo_features.append(pseudo_feature)

    return pseudo_features, pos_samples, neg_samples
def read_docred(file_in, 
                tokenizer, 
                transformer_type="bert",
                max_seq_length=1024, 
                teacher_sig_path="",
                single_results=None,
                collect_srr_entity_types=False,
                entity_type_conflict_policy="majority_first",
                entity_type_conflict_log_dir=""):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    doc_list = []
    total_entities_for_srr = 0
    entity_type_conflicts = []
    if file_in == "":
        return None
    if not docred_rel2id:
        raise FileNotFoundError(
            "dataset/meta/rel2id.json is missing or empty. "
            "Put DocRED rel2id.json under HDER/dataset/meta/ before running."
        )

    with open(file_in, "r") as fh:
        data = json.load(fh)

    if teacher_sig_path != "": # load logits
        basename = os.path.splitext(os.path.basename(file_in))[0]
        attns_file = os.path.join(teacher_sig_path, f"{basename}.attns")
        attns = pickle.load(open(attns_file, 'rb'))

    if single_results is not None:  
        #reorder predictions as relations by title
        pred_pos_samples = 0
        pred_neg_samples = 0
        pred_rels = single_results
        title2preds = {}
        for pred_rel in pred_rels:
            if pred_rel["title"] in title2preds:
                title2preds[pred_rel["title"]].append(pred_rel)
            else:
                title2preds[pred_rel["title"]] = [pred_rel]

    for doc_id in tqdm(range(len(data)), desc="Loading examples"):

        sample = data[doc_id]
        entities = sample['vertexSet']
        entity_type = None
        if collect_srr_entity_types:
            total_entities_for_srr += len(entities)
            entity_type = get_srr_entity_types(
                entities,
                sample.get("title", doc_id),
                conflict_policy=entity_type_conflict_policy,
                conflict_records=entity_type_conflicts,
            )
        entity_start, entity_end = [], []
        # record entities
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((sent_id, pos[0],))
                entity_end.append((sent_id, pos[1] - 1,))

        # add entity markers
        sents, sent_map, sent_pos = add_entity_markers(sample, tokenizer, entity_start, entity_end)

        # training triples with positive examples (entity pairs with labels)
        train_triple = {}

        if "labels" in sample:
            for label in sample['labels']:
                evidence = label['evidence']
                r = int(docred_rel2id[label['r']])

                # update training triples
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r, 'evidence': evidence}]
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r, 'evidence': evidence})
                
        # entity start, end position
        entity_pos = []

        for e in entities:
            entity_pos.append([])
            assert len(e) != 0
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                label = m["type"]
                entity_pos[-1].append((start, end,))

        relations, hts, sent_labels = [], [], []

        for h, t in train_triple.keys(): # for every entity pair with gold relation
            relation = [0] * len(docred_rel2id)
            sent_evi = [0] * len(sent_pos)

            for mention in train_triple[h, t]: # for each relation mention with head h and tail t
                relation[mention["relation"]] = 1
                for i in mention["evidence"]:
                    sent_evi[i] += 1
            relations.append(relation)
            hts.append([h, t])
            sent_labels.append(sent_evi)
            pos_samples += 1

        for h in range(len(entities)):
            for t in range(len(entities)):
                # all entity pairs that do not have relation are treated as negative samples
                if h != t and [h, t] not in hts: #and [t, h] not in hts:
                    relation = [1] + [0] * (len(docred_rel2id) - 1)
                    sent_evi = [0] * len(sent_pos)
                    relations.append(relation)

                    hts.append([h, t])
                    sent_labels.append(sent_evi)
                    neg_samples += 1
        assert len(relations) == len(entities) * (len(entities) - 1)
        if len(sents) > max_seq_length - 2:
            print(f"[WARN] document '{sample.get('title', doc_id)}' has {len(sents)} wordpieces; truncating to {max_seq_length - 2}.")
        sents = sents[:max_seq_length - 2] # truncate, -2 for [CLS] and [SEP]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = build_inputs_with_special_tokens_compat(tokenizer, input_ids)

        base_feature = {'input_ids': input_ids,
                   'entity_pos': entity_pos,
                   'labels': relations,
                   'hts': hts,
                   'sent_pos': sent_pos,
                   'sent_labels': sent_labels,
                   'title': sample['title']
                   }
        if collect_srr_entity_types:
            base_feature['entity_type'] = entity_type
        feature = [base_feature]

        if teacher_sig_path != '': # add evidence distributions from the teacher model
            feature[0]['attns'] = attns[doc_id][:, :len(input_ids)]

        if single_results is not None: # get pseudo documents from predictions of the single run
            offset = 1 if transformer_type in ["bert", "roberta"] else 0
            if sample["title"] in title2preds:
                feature, pos_sample, neg_sample, = get_pseudo_features(feature[0], title2preds[sample["title"]], entities, sent_map, offset, tokenizer)
                pred_pos_samples += pos_sample
                pred_neg_samples += neg_sample

        i_line += len(feature)
        features.extend(feature)
    print("# of documents {}.".format(i_line))
    if single_results is not None:
        print("# of positive examples {}.".format(pred_pos_samples))
        print("# of negative examples {}.".format(pred_neg_samples))

    else:        
        print("# of positive examples {}.".format(pos_samples))
        print("# of negative examples {}.".format(neg_samples))
    if collect_srr_entity_types:
        print_entity_type_conflict_summary(
            file_in,
            entity_type_conflict_policy,
            total_entities_for_srr,
            entity_type_conflicts,
            entity_type_conflict_log_dir,
        )

    return features

