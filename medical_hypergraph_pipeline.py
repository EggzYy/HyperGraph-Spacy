from __future__ import annotations

import json, pathlib, itertools, logging, hashlib
from collections import defaultdict
from typing import Dict, List, Any, Union, Optional, Callable, Tuple
from functools import partial
import re

import spacy, networkx as nx, hypernetx.classes.hypergraph as hnx
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.util import registry

# medspaCy / sciSpacy Components
from medspacy.context import ConText as ConTextComponent
from medspacy.target_matcher import TargetRule, TargetMatcher
from medspacy.section_detection import Sectionizer, SectionRule
from medspacy.context.context_rule import ConTextRule
# PyRuSH is no longer directly imported, as we use its registered name
from scispacy.linking import EntityLinker
from medspacy.custom_tokenizer import create_medspacy_tokenizer

# Ensure custom attributes are registered globally before use
# This prevents errors if a document doesn't trigger a component that sets an attribute.
for attr in (
    "temporality", "experiencer", "section_category", "is_negated",
    "modifiers", "target_rule", "context_rule", "kb_ents", "context_graph"
):
    if not Span.has_extension(attr):
        Span.set_extension(attr, default=None, force=True)
    if not Doc.has_extension(attr): # context_graph is on the Doc
        Doc.set_extension(attr, default=None, force=True)

# Setup structured logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _get_incidence_graph(H: hnx.Hypergraph) -> nx.Graph:
    """Helper to get the incidence graph from a HyperNetX object, supporting multiple versions."""
    if hasattr(H, "incidence_graph"):
        return H.incidence_graph
    if callable(getattr(H, "bipartite", None)):
        return H.bipartite()
    raise RuntimeError("Unsupported HyperNetX version: no incidence graph accessor found.")

PIPELINE_INSTANCE_CONTEXT = {"current": None}

@Language.factory("dynamic_target_rules_component")
def create_dynamic_target_rules_component(nlp, name: str):
    """
    This factory creates the DynamicTargetRulesComponent. It retrieves the parent
    pipeline instance from a context and injects its specific dependencies into the component.
    """
    pipeline_instance = PIPELINE_INSTANCE_CONTEXT["current"]
    if pipeline_instance is None:
        raise ValueError("The pipeline instance context was not set correctly.")

    return DynamicTargetRulesComponent(
        rule_cache=pipeline_instance.rule_cache,
        key_for_rule_func=pipeline_instance._key_for_rule,
        nlp_instance=nlp,
    )

@Language.component("debug_context_graph")
def debug_context_graph(doc):
    # Warn if any modifier has zero edges
    if doc._.context_graph:
        lonely = [m for m in doc._.context_graph.modifiers
                  if not any(m is e[1] for e in doc._.context_graph.edges)]
        if lonely:
            logging.warning(f"{len(lonely)} modifier(s) still have no targets: "
                            f"{[doc[s:e+1].text for m in lonely for s,e in [m.modifier_span]]}")
    return doc

class MedicalHypergraphPipeline:
    """
    A class to encapsulate the medical text processing pipeline, from NLP extraction
    to hypergraph construction.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the pipeline with a given configuration.

        Args:
            config: A dictionary containing configuration parameters like model names
                    and file paths for various rules.
        """
        self.config = config
        self.rule_cache: Dict[str, TargetRule] = self._load_rules_from_file()
        self.context_rule_cache: Dict[str, ConTextRule] = self._load_context_rules_from_file()
        PIPELINE_INSTANCE_CONTEXT["current"] = self
        self.nlp = self._setup_nlp_pipeline()
        PIPELINE_INSTANCE_CONTEXT["current"] = None

    # --------------------------------------------------------------------------
    # Pipeline Setup
    # --------------------------------------------------------------------------

    def _setup_nlp_pipeline(self) -> Language:
        """Loads the spaCy model and configures all pipeline components."""
        model_name = self.config.get("model_name", "en_core_sci_scibert")
        try:
            nlp = spacy.load(model_name)
        except OSError:
            logging.info(f"Model '{model_name}' not found. Downloading...")
            spacy.cli.download(model_name)
            nlp = spacy.load(model_name)

        # 1. Tokenizer and Sentence Splitter (PyRuSH)
        tokenizer = create_medspacy_tokenizer(nlp)
        nlp.tokenizer = tokenizer

        rush_rules_path = self.config.get("rush_rules_path")
        if rush_rules_path and pathlib.Path(rush_rules_path).exists():
            nlp.add_pipe(
                "medspacy_pyrush",
                config={"rules_path": rush_rules_path},
                first=True
            )
            logging.info(f"Added PyRuSH sentencizer with rules from: {rush_rules_path}")
        else:
            logging.warning("`rush_rules_path` not found or file does not exist. Using default spaCy sentencizer.")

        # 2. Sectionizer - Add pipe with no rules, then add them programmatically
        sectionizer = nlp.add_pipe("medspacy_sectionizer", config={"rules": None}, before="ner")
        section_rules = self._load_section_rules_from_file()
        if section_rules:
            sectionizer.add(section_rules)
        else:
            logging.warning("No Sectionizer rules loaded. The sectionizer may not function as expected.")

        # 3. Dynamic Rule Component (Factory pattern to avoid globals)
        nlp.add_pipe("dynamic_target_rules_component", after="ner")

        # 4. Target Matcher
        nlp.add_pipe(
            "medspacy_target_matcher",
            after="dynamic_target_rules_component",
            config={"rules": None}
        )
        target_matcher = nlp.get_pipe("medspacy_target_matcher")
        initial_target_rules = list(self.rule_cache.values())
        if initial_target_rules:
            target_matcher.add(initial_target_rules)
            logging.info(f"Programmatically added {len(initial_target_rules)} rule(s) to the TargetMatcher.")

        # 5. ConText Component - Add pipe with no rules, then add them from the cache
        context_pipe = nlp.add_pipe("medspacy_context", config={"rules": None}, last=True)
        context_pipe.target_span_groups = ["ents", "medspacy_spans"]
        context_pipe.context_graph = True  # type: ignore
        initial_context_rules = list(self.context_rule_cache.values())
        if initial_context_rules:
            context_pipe.add(initial_context_rules)
        else:
            logging.warning("No ConText rules loaded. The context component may not function as expected.")
        
        nlp.add_pipe("debug_context_graph", after="medspacy_context")

        # 6. Entity Linker
        linker_config = self.config.get("linker_config", {
            "resolve_abbreviations": True, "linker_name": "umls", "threshold": 0.7
        })
        nlp.add_pipe("scispacy_linker", config=linker_config, last=True)

        logging.info(f"Pipeline created successfully with components: {nlp.pipe_names}")
        return nlp

    # --------------------------------------------------------------------------
    # Rule Management
    # --------------------------------------------------------------------------

    def _load_rules_from_file(self) -> Dict[str, TargetRule]:
        """Loads TargetRules from the JSON file specified in the config."""
        rules_path_str = self.config.get("target_rules_path")
        if not rules_path_str:
            logging.info("`target_rules_path` not in config. Starting with an empty rule set.")
            return {}
        
        rules_path = pathlib.Path(rules_path_str)
        rule_cache = {}

        if not rules_path.exists():
            logging.info(f"No saved TargetRules found at {rules_path}—starting with an empty rule set.")
            return rule_cache

        try:
            with rules_path.open("r", encoding="utf8") as f:
                raw_rules_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Could not read or parse {rules_path}: {e}. Starting fresh.")
            return rule_cache

        if not isinstance(raw_rules_data, list):
            logging.warning(f"Expected a list of rules in {rules_path}. Starting fresh.")
            return rule_cache

        loaded_rules = [self._coerce_saved_rule(item) for item in raw_rules_data]
        good_rules = [rule for rule in loaded_rules if rule and self._is_good_rule(rule)]

        discarded_count = len(raw_rules_data) - len(good_rules)
        if discarded_count > 0:
            logging.warning(f"Discarded {discarded_count} invalid rule(s) from {rules_path}")

        for rule in good_rules:
            rule_cache[self._key_for_rule(rule)] = rule

        logging.info(f"Restored {len(rule_cache)} TargetRules from {rules_path}")
        return rule_cache

    def _load_section_rules_from_file(self) -> List[SectionRule]:
        """Loads SectionRule objects from the JSON file specified in the config."""
        rules_path_str = self.config.get("section_rules_path")
        if not rules_path_str:
            logging.warning("`section_rules_path` not specified in config. No section rules will be loaded.")
            return []
        
        rules_path = pathlib.Path(rules_path_str)
        if not rules_path.exists():
            logging.warning(f"Sectionizer rules file not found at {rules_path}. No section rules will be loaded.")
            return []

        try:
            with rules_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            raw_rules = data.get("section_rules", [])
            if not isinstance(raw_rules, list):
                logging.error(f"Sectionizer rules file {rules_path} must contain a list under the 'section_rules' key.")
                return []
            
            section_rules = [SectionRule.from_dict(rule_data) for rule_data in raw_rules]
            logging.info(f"Loaded {len(section_rules)} Sectionizer rules from {rules_path}")
            return section_rules
        except (json.JSONDecodeError, IOError, KeyError, TypeError) as e:
            logging.error(f"Failed to load or parse Sectionizer rules from {rules_path}: {e}")
            return []

    def _load_context_rules_from_file(self) -> Dict[str, ConTextRule]:
        """Loads ConTextRule objects from JSON and populates the cache."""
        rules_path_str = self.config.get("context_rules_path")
        if not rules_path_str:
            logging.warning("`context_rules_path` not specified in config. Starting with empty context rule set.")
            return {}

        rules_path = pathlib.Path(rules_path_str)
        if not rules_path.exists():
            logging.warning(f"Context rules file not found at {rules_path}. Starting fresh.")
            return {}

        cache = {}
        try:
            with rules_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            raw_rules = data.get("context_rules", [])
            if not isinstance(raw_rules, list):
                logging.error(f"Context rules file {rules_path} must contain a list under 'context_rules' key.")
                return {}
            
            for rule_data in raw_rules:
                rule = ConTextRule.from_dict(rule_data)
                key = self._key_for_context_rule(rule)
                cache[key] = rule
            
            logging.info(f"Loaded and cached {len(cache)} ConText rules from {rules_path}")
            return cache
        except (json.JSONDecodeError, IOError, KeyError, TypeError) as e:
            logging.error(f"Failed to load or parse ConText rules from {rules_path}: {e}")
            return {}

    def save_rules(self) -> None:
        """Serializes all valid TargetRules from the cache to a JSON file."""
        rules_path = pathlib.Path(self.config["target_rules_path"])
        valid_rules_to_save = [
            self._rule_to_dict(rule) for rule in self.rule_cache.values() if self._is_good_rule(rule)
        ]

        if not valid_rules_to_save:
            logging.info(f"No valid target rules in cache to save to {rules_path}.")
            return

        with rules_path.open("w", encoding="utf8") as f:
            json.dump(valid_rules_to_save, f, indent=2)
        logging.info(f"Saved {len(valid_rules_to_save)} TargetRule(s) to {rules_path}")
        
    def save_context_rules(self) -> None:
        """Serializes all ConTextRules from the cache to a JSON file."""
        rules_path_str = self.config.get("context_rules_path")
        if not rules_path_str:
            logging.error("Cannot save context rules: `context_rules_path` not specified in config.")
            return

        rules_path = pathlib.Path(rules_path_str)
        rules_to_save = [rule.to_dict() for rule in self.context_rule_cache.values()]

        if not rules_to_save:
            logging.info(f"No context rules in cache to save to {rules_path}.")
            return
            
        # Wrap in the expected dictionary structure
        output_data = {"context_rules": rules_to_save}
            
        with rules_path.open("w", encoding="utf8") as f:
            json.dump(output_data, f, indent=2)
        logging.info(f"Saved {len(rules_to_save)} ConTextRule(s) to {rules_path}")

    @staticmethod
    def _key_for_rule(rule: TargetRule) -> str:
        """Generates a unique key for a TargetRule."""
        key_parts: List[str] = []
        if rule.pattern:
            if isinstance(rule.pattern, str):
                key_parts.append(f"pattern_regex:{rule.pattern}")
            elif isinstance(rule.pattern, list):
                key_parts.append(f"pattern_tokens:{json.dumps(rule.pattern, sort_keys=True)}")
        else:
            key_parts.append(f"literal_exact:{rule.literal.lower()}")
        key_parts.append(f"category:{rule.category.lower()}")
        return "|".join(key_parts)
        
    @staticmethod
    def _key_for_context_rule(rule: ConTextRule) -> str:
        """Generates a unique key for a ConTextRule."""
        key_parts: List[str] = []
        if rule.pattern:
            key_parts.append(f"pattern_tokens:{json.dumps(rule.pattern, sort_keys=True)}")
        else:
            key_parts.append(f"literal_exact:{rule.literal.lower()}")
        key_parts.append(f"category:{rule.category.lower()}")
        key_parts.append(f"direction:{rule.direction.lower()}")
        return "|".join(key_parts)

    @staticmethod
    def _is_good_rule(obj: Any) -> bool:
        return (isinstance(obj, TargetRule) and
                obj.literal and obj.literal.strip() and
                obj.category and obj.category.strip())

    @staticmethod
    def _rule_to_dict(rule: TargetRule) -> dict[str, Any]:
        return {"literal": rule.literal, "pattern": rule.pattern, "category": rule.category, "attributes": rule.attributes}

    def _dict_to_rule(self, data: dict[str, Any]) -> TargetRule:
        try:
            return TargetRule(
                literal=data["literal"],
                category=data["category"],
                pattern=data.get("pattern"),
                attributes=data.get("attributes"),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Invalid dict for TargetRule: {data}. Error: {e}")

    def _coerce_saved_rule(self, obj: Any) -> Optional[TargetRule]:
        try:
            if isinstance(obj, dict):
                rule = self._dict_to_rule(obj)
                return rule if self._is_good_rule(rule) else None
        except ValueError as e:
            logging.warning(f"Skipping invalid rule data: {obj}. Error: {e}")
        return None
        
    def add_context_rule(self, literal: str, category: str, direction: str, pattern: Optional[List[Dict]] = None) -> None:
        """
        Adds a new ConTextRule to the pipeline dynamically.
        
        The rule is added to the cache and the active pipeline component.
        Call `save_context_rules()` to persist the new rule to the JSON file.

        Args:
            literal: The literal phrase of the rule.
            category: The category of the modifier (e.g., "NEGATED_EXISTENCE").
            direction: The direction of the rule (e.g., "FORWARD", "BACKWARD").
            pattern: An optional spaCy pattern for more complex matches.
        """
        new_rule = ConTextRule(literal=literal, category=category, direction=direction, pattern=pattern)
        rule_key = self._key_for_context_rule(new_rule)

        if rule_key in self.context_rule_cache:
            logging.info(f"ConTextRule for '{literal}' already exists. Skipping.")
            return

        # Add to cache
        self.context_rule_cache[rule_key] = new_rule
        
        # Add to the live component in the nlp pipeline
        try:
            context_pipe = self.nlp.get_pipe("medspacy_context")
            context_pipe.add([new_rule])
            logging.info(f"Dynamically added new ConTextRule: '{literal}' | {category} | {direction}")
        except KeyError:
            logging.error("Could not find 'medspacy_context' in the pipeline to add new rule.")

    # --------------------------------------------------------------------------
    # Core Processing
    # --------------------------------------------------------------------------

    def process_text(self, text: str) -> tuple[hnx.Hypergraph, dict[str, Any]]:
        """
        Processes a string of medical text to produce a hypergraph and its atom data.

        Args:
            text: The medical text to process.

        Returns:
            A tuple containing:
            - The generated HyperNetX.Hypergraph object.
            - The JSON-serializable dictionary of hypergraph atoms.
        """
        doc = self.nlp(text)
        hypergraph_data = self._extract_hypergraph_atoms(doc)
        hypergraph = self.build_hypergraph(hypergraph_data, doc)
        return hypergraph, hypergraph_data

    def _extract_hypergraph_atoms(self, doc: Doc) -> Dict[str, Any]:
        """Extracts structured "atoms" from a spaCy Doc for hypergraph construction."""
        linker = self.nlp.get_pipe("scispacy_linker")
        
        entity_atoms, _ = self._create_entity_instance_atoms(doc, linker)
        relation_hints = self._create_relation_hints(doc, entity_atoms)
        context_graph_data = self._serialize_context_graph(doc, get_node_id=self._get_robust_node_id_factory(doc))

        return {
            "entity_atoms": entity_atoms,
            "sentence_relation_hints": relation_hints,
            "context_graph": context_graph_data,
        }

    def _get_robust_node_id_factory(self, doc: Doc) -> Callable[[Any], str]:
        """Creates a closure for generating stable node IDs for a given doc."""
        doc_hash = hashlib.md5(doc.text.encode()).hexdigest()[:8]
        def get_robust_node_id(node_obj: Any) -> str:
            if isinstance(node_obj, Span):
                return f"span_{doc_hash}_{node_obj.start_char}_{node_obj.end_char}"
            else:  # ConTextModifier-like object
                tok_start, tok_end = node_obj.modifier_span
                span = doc[tok_start : tok_end + 1]          # end is inclusive
                return f"span_{doc_hash}_{span.start_char}_{span.end_char}"
        return get_robust_node_id

    def _create_entity_instance_atoms(self, doc: Doc, linker: EntityLinker) -> tuple[List[dict], dict]:
        """
        Creates an atom for every INSTANCE of an entity, preserving all context.
        It also returns a map of canonical concept IDs.
        """
        instance_atoms = []
        canonical_id_map = {}
        canonical_id_counter = itertools.count(1)

        # Get all unique entity spans from NER and the target matcher.
        all_ents = list(doc.ents) + doc.spans.get("medspacy_spans", [])
        unique_spans = []
        seen_spans = set()
        for s in sorted(all_ents, key=lambda sp: (sp.start_char, sp.end_char)):
            if (s.start_char, s.end_char) not in seen_spans:
                unique_spans.append(s)
                seen_spans.add((s.start_char, s.end_char))

        # For each unique span, create a unique instance atom.
        for i, ent_span in enumerate(unique_spans):
            # 1. Determine the CANONICAL ID for the concept (e.g., all "GGT"s are 'ent_10')
            surf = ent_span.text.lower().strip()
            best_cui = ent_span._.kb_ents[0][0] if ent_span._.kb_ents else None
            canonical_key = (surf, best_cui)

            if canonical_key not in canonical_id_map:
                canonical_id_map[canonical_key] = f"ent_{next(canonical_id_counter)}"
            canonical_id = canonical_id_map[canonical_key]

            # 2. Create the INSTANCE ATOM with its own unique context.
            section = ent_span._.section
            instance_atoms.append({
                "instance_id": f"inst_{i}",
                "canonical_id": canonical_id,
                "text": ent_span.text,
                "label": ent_span.label_,
                "start_char": ent_span.start_char,
                "end_char": ent_span.end_char,
                "sentence_text": ent_span.sent.text.strip().replace("\n", " "),
                "section_title": section.title if section else "UNCATEGORIZED",
                "umls_linking": self._get_umls_details(ent_span, linker),
                "contextual_modifiers": self._get_context_modifiers(ent_span, doc),
            })
        return instance_atoms, canonical_id_map

    def _get_umls_details(self, ent: Span, linker: EntityLinker) -> List[dict]:
        """Extracts UMLS linking information for an entity."""
        return [
            {
                "cui": cui, "score": round(score, 3),
                "definition": getattr(linker.kb.cui_to_entity.get(cui), "definition", None),
                "aliases": getattr(linker.kb.cui_to_entity.get(cui), "aliases", [])[:5],
                "snomed_ids": getattr(linker.kb.cui_to_entity.get(cui), "db_ids", {}).get("SNOMEDCT_US", []),
            } for cui, score in ent._.kb_ents
        ]

    def _get_context_modifiers(self, ent: Span, doc: Doc) -> List[dict]:
        """Extracts ConText modifier information for an entity."""
        ctx_mods = []
        for mod in ent._.modifiers:
            try:
                trigger_span = doc[mod.modifier_span[0] : mod.modifier_span[1] + 1]
                scope_span_text = ""
                if mod.scope_span:
                    scope_span = doc[mod.scope_span[0] : mod.scope_span[1] + 1]
                    scope_span_text = scope_span.text

                ctx_mods.append({
                    "category": mod.category, "trigger_phrase": trigger_span.text,
                    "direction": mod.direction, "scope_phrase": scope_span_text,
                })
            except IndexError:
                logging.warning(f"Could not resolve modifier span for '{ent.text}'. Skipping modifier.")
        return ctx_mods

    def _create_relation_hints(self, doc: Doc, entity_atoms: List[dict]) -> List[dict]:
        """
        Generates sentence-level co-occurrence and uncertainty hints.
        This version correctly uses the instance-based data model.
        """
        relation_hints = []
        uncertainty_labels = {"UNCERTAINTY", "NEGATED_EXISTENCE", "POSSIBLE_EXISTENCE", "HYPOTHETICAL"}

        for i, sent in enumerate(doc.sents, start=1):
            sent_ents = [
                ea for ea in entity_atoms
                if sent.start_char <= ea["start_char"] < sent.end_char
            ]
            if not sent_ents:
                continue

            ent_instance_ids = [ea["instance_id"] for ea in sent_ents]

            triggers_with_categories = {
                (mod["trigger_phrase"], mod["category"])
                for ea in sent_ents
                for mod in ea["contextual_modifiers"] if mod["category"] in uncertainty_labels
            }

            if triggers_with_categories:
                relation_hints.append({
                    "atom_type": "probabilistic_relation_hint",
                    "id": f"sent_{i}",
                    "sentence_text": sent.text.strip().replace("\n", " "),
                    "start_char": sent.start_char,
                    "end_char": sent.end_char,
                    "entities_involved_ids": ent_instance_ids,
                    "uncertainty_triggers": [
                        {"trigger": trigger, "category": category}
                        for trigger, category in sorted(triggers_with_categories)
                    ],
                })
        return relation_hints

    def _serialize_context_graph(self, doc: Doc, get_node_id: Callable[[Any], str]) -> Dict[str, Any]:
        """Serializes the ConText graph with robust, stable IDs."""
        cg = doc._.context_graph
        if not cg:
            return {"targets": [], "modifiers": [], "edges": []}

        def serialize_node(node_obj: Any) -> Dict[str, Any]:
            """Helper to serialize a target or modifier node."""
            if isinstance(node_obj, Span):
                span = node_obj
            else:  # ConTextModifier
                start, end = node_obj.modifier_span
                span = doc[start : end + 1]

            return {
                "id": get_node_id(node_obj), "text": span.text,
                "category": self._span_category(node_obj),
                "start_char": span.start_char, "end_char": span.end_char,
            }

        return {
            "targets": [serialize_node(t) for t in cg.targets],
            "modifiers": [serialize_node(m) for m in cg.modifiers],
            "edges": [{"target_id": get_node_id(t), "modifier_id": get_node_id(m)}
                      for t, m in cg.edges],
        }

    @staticmethod
    def _span_category(span: Any) -> Optional[str]:
        """Robustly gets a category from a spaCy/medspaCy object."""
        if hasattr(span, "category"): return span.category
        rule = getattr(span._, "target_rule", None) or getattr(span._, "context_rule", None)
        if rule: return rule.category
        return getattr(span, "label_", None)

    # --------------------------------------------------------------------------
    # Hypergraph Construction and Persistence
    # --------------------------------------------------------------------------

    @staticmethod
    def build_hypergraph(hypergraph_data: Dict[str, Any], doc: Doc) -> hnx.Hypergraph:
        """
        Builds a comprehensive hypergraph using the new instance-based atom model.
        """
        instance_atoms = hypergraph_data["entity_atoms"]

        node_attrs = {}
        for inst in instance_atoms:
            if inst["canonical_id"] not in node_attrs:
                node_attrs[inst["canonical_id"]] = {
                    'text': inst["text"],
                    'label': inst["label"],
                    'umls': inst["umls_linking"]
                }

        cg_data = hypergraph_data["context_graph"]
        for mod in cg_data["modifiers"]:
            if mod["id"] not in node_attrs:
                node_attrs[mod["id"]] = {'text': mod['text'], 'label': 'MODIFIER', 'category': mod['category']}

        edge_members = defaultdict(set)
        edge_attrs = {}

        uncertain_spans = {(h['start_char'], h['end_char']) for h in hypergraph_data["sentence_relation_hints"]}
        for i, sent in enumerate(doc.sents, start=1):
            sent_canonical_ids = {
                inst["canonical_id"] for inst in instance_atoms
                if sent.start_char <= inst["start_char"] < sent.end_char
            }
            if not sent_canonical_ids: continue

            edge_id = f"sent_{i}"
            edge_members[edge_id] = sent_canonical_ids
            is_uncertain = (sent.start_char, sent.end_char) in uncertain_spans
            edge_attrs[edge_id] = {"certainty": 0.5 if is_uncertain else 1.0, "text": sent.text.strip().replace("\n", " "), "level": "sentence"}

        all_section_titles = {inst["section_title"] for inst in instance_atoms}
        if any(title != "UNCATEGORIZED" for title in all_section_titles):
            section_to_ids = defaultdict(set)
            for inst in instance_atoms:
                if (title := inst["section_title"]) != "UNCATEGORIZED":
                    section_to_ids[title].add(inst["canonical_id"])
            for i, (title, ids) in enumerate(section_to_ids.items(), 1):
                edge_members[f"sec_{i}"] = ids
                edge_attrs[f"sec_{i}"] = {"certainty": 1.0, "text": f"Section: {title}", "level": "section"}
        else:
             paragraphs = [p.strip() for p in doc.text.split('\n\n') if p.strip()]
             current_pos = 0
             for i, para_text in enumerate(paragraphs):
                para_start = doc.text.find(para_text, current_pos)
                para_end = para_start + len(para_text)
                current_pos = para_end
                para_ids = {inst["canonical_id"] for inst in instance_atoms if para_start <= inst["start_char"] < para_end}
                if para_ids:
                    edge_members[f"para_{i+1}"] = para_ids
                    edge_attrs[f"para_{i+1}"] = {"certainty": 1.0, "text": f"Paragraph {i+1}", "level": "section"}

        missing_map: list[tuple[str, str, str]] = []
        doc_hash = hashlib.md5(doc.text.encode()).hexdigest()[:8]
        span_to_canonical_id = {
            f"span_{doc_hash}_{inst['start_char']}_{inst['end_char']}": inst['canonical_id']
            for inst in instance_atoms
        }
        for i, edge in enumerate(cg_data["edges"]):
            target_span_id, modifier_id = edge["target_id"], edge["modifier_id"]

            canonical_target_id = span_to_canonical_id.get(target_span_id)

            if not canonical_target_id:
                try:
                    target_start_char = int(target_span_id.split('_')[-2])
                    target_end_char = int(target_span_id.split('_')[-1])
                    for span_id, can_id in span_to_canonical_id.items():
                        span_start = int(span_id.split('_')[-2])
                        span_end = int(span_id.split('_')[-1])
                        if abs(span_start - target_start_char) < 5 and abs(span_end - target_end_char) < 5:
                            canonical_target_id = can_id
                            logging.debug(f"Fuzzy matched target_span_id {target_span_id} to {span_id}")
                            break
                except (ValueError, IndexError):
                     logging.warning(f"Could not parse span ID for fuzzy matching: {target_span_id}")


            if canonical_target_id and modifier_id in node_attrs:
                edge_members[f"ctx_{i}"] = {canonical_target_id, modifier_id}
                edge_attrs[f"ctx_{i}"] = {"level": "contextual_link", "certainty": 1.0}
            else:
                reason = "No canonical ID" if not canonical_target_id else "Modifier ID not in node_attrs"
                missing_map.append((target_span_id, modifier_id, reason))

        if missing_map:
            logging.warning(
                f"Dropped {len(missing_map)} ConText edges. Details: {missing_map[:5]}"
            )

        nodes_in_an_edge = {node for edge_set in edge_members.values() for node in edge_set}
        for node_id in node_attrs:
            if node_id not in nodes_in_an_edge:
                edge_id = f"loop_{node_id}"
                edge_members[edge_id] = {node_id}
                edge_attrs[edge_id] = {"level": "loop_span", "certainty": 1.0}

        H = hnx.Hypergraph(edge_members)
        for node_id, attrs in node_attrs.items():
            if node_id in H.nodes:
                if H.nodes[node_id].attrs is None: H.nodes[node_id].attrs = {}
                H.nodes[node_id].attrs.update(attrs)

        inc_g = _get_incidence_graph(H)
        for eid, attrs in edge_attrs.items():
            if eid in H.edges:
                if H.edges[eid].attrs is None: H.edges[eid].attrs = {}
                H.edges[eid].attrs.update(attrs)
                for n in H.edges[eid].elements:
                    if (n, eid) in inc_g.edges: inc_g.edges[(n, eid)].update(attrs)

        for n_id in H.nodes:
            if H.nodes[n_id].attrs is None: H.nodes[n_id].attrs = {}
            if n_id in H.incidence_dict:
                certs = [edge_attrs[eid].get("certainty", 1.0) for eid in H.incidence_dict[n_id] if eid in edge_attrs]
                H.nodes[n_id].attrs["certainty"] = min(certs) if certs else 1.0
            else:
                H.nodes[n_id].attrs["certainty"] = 1.0

        return H

    @staticmethod
    def save_hypergraph(H: hnx.Hypergraph, path: Union[str, pathlib.Path]) -> None:
        """Serializes the hypergraph to a gpickle file in a version-agnostic way."""
        inc_g = _get_incidence_graph(H).copy()
        nx.set_node_attributes(inc_g, False, "is_hyperedge")

        for node_id in H.nodes:
            node_obj = H.nodes[node_id]
            if node_obj.attrs:
                inc_g.nodes[node_id]["node_attrs"] = dict(node_obj.attrs)

        for edge_id in H.edges:
            edge_obj = H.edges[edge_id]
            if edge_id in inc_g:
                inc_g.nodes[edge_id]["is_hyperedge"] = True
                if edge_obj.attrs:
                    inc_g.nodes[edge_id]["edge_attrs"] = dict(edge_obj.attrs)

        nx.write_gpickle(inc_g, str(path))
        logging.info(f"Hypergraph incidence graph saved to {pathlib.Path(path).resolve()}")

    @staticmethod
    def load_hypergraph(path: Union[str, pathlib.Path]) -> hnx.Hypergraph:
        """Loads a hypergraph from a gpickle file."""
        inc_g = nx.read_gpickle(str(path))

        if hasattr(hnx.Hypergraph, "from_bipartite_graph"):
            H = hnx.Hypergraph.from_bipartite_graph(inc_g)
        elif hasattr(hnx.Hypergraph, "from_incidence_graph"):
            H = hnx.Hypergraph.from_incidence_graph(inc_g)
        else:
            edge_members = defaultdict(set)
            for u, v in inc_g.edges:
                if inc_g.nodes[u].get("is_hyperedge"):
                    edge_members[u].add(v)
                elif inc_g.nodes[v].get("is_hyperedge"):
                    edge_members[v].add(u)
            H = hnx.Hypergraph(edge_members)

        for edge_id in H.edges:
            edge_obj = H.edges[edge_id]
            attrs = inc_g.nodes[edge_id].get("edge_attrs")
            if attrs:
                if edge_obj.attrs is None: edge_obj.attrs = {}
                edge_obj.attrs.update(attrs)

        for node_id in H.nodes:
            node_obj = H.nodes[node_id]
            attrs = inc_g.nodes[node_id].get("node_attrs")
            if attrs:
                if node_obj.attrs is None: node_obj.attrs = {}
                node_obj.attrs.update(attrs)

        logging.info(f"Hypergraph loaded successfully from {pathlib.Path(path).resolve()}")
        return H


class DynamicTargetRulesComponent:
    """A spaCy pipeline component to dynamically add TargetRules from NER entities."""
    def __init__(
        self,
        rule_cache: Dict[str, TargetRule],
        key_for_rule_func: Callable[[TargetRule], str],
        nlp_instance: Language,
    ):
        self.rule_cache = rule_cache
        self.key_for_rule = key_for_rule_func
        self.nlp = nlp_instance

    def __call__(self, doc: Doc) -> Doc:
        """
        Identifies new potential rules from NER entities and adds them to the
        TargetMatcher for the current `doc` processing.
        """
        target_matcher = self.nlp.get_pipe("medspacy_target_matcher")
        newly_added_rules = []

        for ent in doc.ents:
            literal = ent.text.strip()
            if not literal: continue

            new_rule = TargetRule(literal, ent.label_)
            new_key = self.key_for_rule(new_rule)

            if new_key not in self.rule_cache:
                self.rule_cache[new_key] = new_rule
                newly_added_rules.append(new_rule)

        if newly_added_rules:
            target_matcher.add(newly_added_rules)
            logging.info(f"Dynamically added {len(newly_added_rules)} new rule(s) to TargetMatcher.")

        return doc