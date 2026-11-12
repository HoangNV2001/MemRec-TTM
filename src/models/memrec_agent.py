"""
MemRec Agent Model v2
Three-stage architecture: Stage-R (Retrieval) + Stage-ReRank + Stage-W (Write)
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
import json
import time

from src.memory import (
    UserItemGraph,
    NeighborPruner,
    SnippetPacker,
    MemRecManager,
    MemoryStorage
)
from src.memory.pruner_llm_rules import LLMRulePruner
from src.memory.ttm import TrajectoryRetriever
from src.memory.semantic import SemanticEvidence
from src.memory.trend import TrendEvidence
from src.models.reranker_llm import LLMReranker
from src.models.reranker_vector import VectorReranker


class MemRecAgent:
    """
    MemRec Agent: Three-stage collaborative memory
    
    Workflow (per interaction):
    1. Pruning: Select top-k neighbors (mixing constraint)
    2. Packing: Pack snippets within token budget
    3. Stage-R: Generate facets, vector_profile, support_edges (no candidate scoring)
    4. Stage-ReRank: Score candidates using reranker (LLM or vector)
    5. (Optional) Stage-W: Generate user_patches + neighbor_patches after feedback
    """
    
    def __init__(
        self,
        dataset,
        llm_client,
        k: int = 16,
        tau: int = 1800,
        n_facets: int = 7,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        mix_min_users: int = 4,
        mix_min_items: int = 6,
        fanout_cap: int = 6,
        reranker_mode: str = "vector",  # "vector" or "llm"
        pruner_mode: str = "llm_rules",  # "hybrid_rule", "learned_mlp", or "llm_rules"
        pruner_checkpoint: Optional[str] = None,
        enable_stage_r: bool = True,  # Ablation: control Stage-R
        reranker_llm_client = None,  # Optional separate LLMClient for reranker
        vanilla: bool = False,  # Baseline: no memory, LLM reranks from recent history only
        vanilla_history_len: int = 10,
        ttm: Optional[Dict] = None,  # MemRec-TTM options (see src/memory/ttm.py), None = original MemRec
        rerank_output: str = 'scores',  # 'scores' (upstream) or 'ranking' (ordered ID list)
        isolated: bool = False,  # baselines: own items only as neighbors (no collaborative user neighbors)
        debug: bool = False
    ):
        """
        Initialize MemRec Agent
        
        Args:
            dataset: RecDataset instance
            llm_client: LLMClient instance (with cache)
            k: Number of neighbors selected by pruner
            tau: Token budget
            n_facets: Number of facets requested
            temperature: LM sampling temperature
            max_tokens: Maximum tokens generated (Stage-R)
            mix_min_users: Minimum user neighbors
            mix_min_items: Minimum item neighbors
            fanout_cap: Stage-W propagation fanout cap
            reranker_mode: Reranker mode ("vector" or "llm")
            pruner_mode: Pruner mode ("hybrid_rule", "learned_mlp", or "llm_rules")
            pruner_checkpoint: Pruner MLP checkpoint path (for learned_mlp)
            debug: Whether to print debug information
        """
        self.dataset = dataset
        self.llm_client = llm_client
        self.k = k
        self.tau = tau
        self.n_facets = n_facets
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.mix_min_users = mix_min_users
        self.mix_min_items = mix_min_items
        self.fanout_cap = fanout_cap
        self.isolated = isolated  # Stage-W then updates only the user and the clicked item (no propagation)
        self.reranker_mode = reranker_mode
        self.pruner_mode = pruner_mode
        self.enable_stage_r = enable_stage_r  # Ablation control
        self.pruner_checkpoint = pruner_checkpoint
        self.vanilla = vanilla
        self.vanilla_history_len = vanilla_history_len
        self.debug = debug
        self.perturb = []  # evaluation-time memory ablation, set by the runner after warm-up (see _perturb)
        
        # Build user-item graph
        print("Building user-item graph...")
        self.graph = UserItemGraph(dataset)
        print(self.graph)
        
        # MemRec-TTM: transition graph + trajectory evidence
        ttm = ttm or {}
        self.ttm = None
        self.ttm_read = self.ttm_evidence = False
        self.k_traj = 0
        if ttm.get('enabled'):
            self.ttm = TrajectoryRetriever(dataset, mode=ttm.get('mode', 'ppr'), m=ttm.get('m', 6),
                                           restart=ttm.get('restart', 0.15))
            self.ttm_read = ttm.get('read', True)
            self.ttm_evidence = ttm.get('rerank_evidence', True)
            self.k_traj = ttm.get('k_traj', 4) if self.ttm_read else 0
            print(f"  MemRec-TTM: {self.ttm.graph.stats()} read={self.ttm_read} (k_traj={self.k_traj}) "
                  f"evidence={self.ttm_evidence} mode={self.ttm.mode}")
        # semantic (dense) retrieval evidence next to each candidate; path injected by the runner
        self.sem = SemanticEvidence(ttm['semantic_emb'], ttm.get('semantic_recent', 10)) \
            if ttm.get('semantic_evidence') else None
        # trend evidence: recent-popularity channel of stage 1; cutoff injected by the runner
        self.trend = TrendEvidence(dataset, ttm['trend_cutoff'], ttm.get('trend_window', 90)) \
            if ttm.get('trend_evidence') else None
        self.recent_k = ttm.get('recent_k', 0)  # C0: recent items shown to the reranker (0 = upstream)
        self.temporal_facets = ttm.get('temporal_facets', False)  # B1: Stage-R reads the timeline, two horizons
        self.timeline_k = ttm.get('timeline_k', 10)
        k_base = k - self.k_traj  # equal neighbor budget: trajectory slots come out of k
        
        # Initialize components
        if pruner_mode == "llm_rules":
            # Use LLM-generated domain-specific rules
            self.pruner = LLMRulePruner(
                dataset_name=dataset.name if hasattr(dataset, 'name') else 'instructrec-books',
                k=k_base,
                dataset=dataset,
                mix_min_users=mix_min_users,
                mix_min_items=mix_min_items,
                isolated=isolated,
            )
            print(f"  Using LLM Rule-Based Pruner for domain-specific neighbor selection")
        else:
            # Use original parametric pruner
            self.pruner = NeighborPruner(
                k=k_base,
                mix_min_users=mix_min_users,
                mix_min_items=mix_min_items,
                mode=pruner_mode,
                checkpoint=pruner_checkpoint
            )
        self.packer = SnippetPacker(tau=tau)
        self.manager = MemRecManager(llm_client)
        self.storage = MemoryStorage()
        
        # Initialize item descriptions from metadata
        if dataset.item_metadata:
            self.storage.initialize_item_descriptions(dataset.item_metadata)
        
        # Initialize Reranker
        if reranker_mode == "llm":
            print("  Reranker: LLM-based")
            # If separate reranker_llm_client is provided, use it; otherwise use default llm_client
            reranker_client = reranker_llm_client if reranker_llm_client is not None else llm_client
            if reranker_llm_client is not None:
                print(f"    Using separate LLMClient: {reranker_client.model} @ {reranker_client.api_endpoint}")
            self.reranker = LLMReranker(reranker_client, output_format=rerank_output)
        else:
            print("  Reranker: Vector-based (fast)")
            self.reranker = VectorReranker()
        
        # Statistics
        self.n_stage_r_calls = 0
        self.n_stage_rr_calls = 0
        self.n_stage_w_calls = 0
    
    def rerank(
        self,
        user_id: int,
        candidates: List[int],
        instruction: str = None,  # User instruction (like iAgent)
        return_details: bool = False,
        debug_logger = None,
        history_items: List[int] = None  # Strict-past history (used by vanilla mode)
    ) -> Tuple[List[int], Dict]:
        """
        Rerank candidate items for user (three stages: Pruning → Stage-R → Stage-ReRank)
        
        Args:
            user_id: User ID
            candidates: List of candidate item IDs (typically 10)
            instruction: User instruction/intent (optional, from dataset)
            return_details: Whether to return detailed information
            debug_logger: Optional debug logger
            
        Returns:
            (ranked_items, details)
            - ranked_items: Reranked list of item IDs (descending order)
            - details: Contains facets, scores, retrieval_bundle, etc.
        """
        if self.vanilla:
            return self._rerank_vanilla(user_id, candidates, instruction, history_items, debug_logger)
        
        # 1. Pruning: Select top-k neighbors (mixing constraint)
        t0 = time.time()
        pruned_subgraph = self.pruner.prune(user_id, self.graph, candidates)
        ttm_state = None
        if self.ttm is not None:
            hist = history_items if history_items is not None else self.dataset.get_user_train_items(user_id)
            ttm_state = self.ttm.user_state(hist)
            if self.ttm_read:
                taken = {(n['type'], n['id']) for n in pruned_subgraph.get('neighbors', [])}
                traj = [n for n in self.ttm.trajectory_neighbors(ttm_state, self.k_traj + 4)
                        if ('item', n['id']) not in taken][:self.k_traj]
                pruned_subgraph['neighbors'] = traj + pruned_subgraph.get('neighbors', [])
                pruned_subgraph['n_items'] = pruned_subgraph.get('n_items', 0) + len(traj)
        pruning_time = time.time() - t0
        
        # 2. Packing: Pack within token budget
        user_memory_summary = self.storage.render_user_summary(user_id)
        packed_context = self.packer.pack(
            pruned_subgraph=pruned_subgraph,
            dataset=self.dataset,
            candidates=candidates,
            user_memory_summary=user_memory_summary
        )
        
        if self.debug:
            print(f"\n[MemRec] User {user_id}")
            print(f"  Pruned: {pruned_subgraph['n_items']} items, {pruned_subgraph['n_users']} users")
            print(f"  Packed: {packed_context['n_neighbors']} neighbors, ~{packed_context['estimated_tokens']} tokens")
        
        # 3. Stage-R: Retrieval (generate facets + vector_profile, no candidate scoring)
        # Ablation: If enable_stage_r=False, skip Stage-R
        if self.enable_stage_r:
            t0 = time.time()
            timeline_text = None
            if self.temporal_facets:
                hist = history_items if history_items is not None else self.dataset.get_user_train_items(user_id)
                timeline_text = self._timeline_text(user_id, hist)
            retrieval_bundle = self.manager.run_stage_r(
                user_id=user_id,
                packed_context=packed_context,
                n_facets=self.n_facets,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                debug_logger=debug_logger,
                timeline_text=timeline_text
            )
            stage_r_time = time.time() - t0
            self.n_stage_r_calls += 1
            
            if self.debug:
                print(f"  Stage-R: {len(retrieval_bundle.get('facets', []))} facets, {stage_r_time*1000:.1f}ms")
        else:
            # No-Mem mode: empty retrieval_bundle
            retrieval_bundle = {
                'facets': [],
                'vector_profile': None,
                'support_edges': []
            }
            stage_r_time = 0.0
            
            if self.debug:
                print(f"  Stage-R: SKIPPED (No-Mem mode)")
        
        # 4. Stage-ReRank: Score candidates
        t0 = time.time()
        candidate_list = self._prepare_candidate_list(candidates)
        if self.ttm_evidence and ttm_state is not None:
            evidence = self.ttm.candidate_evidence(ttm_state, candidates)
            for c in candidate_list:
                c['evidence'] = evidence[c['id']]
        self._add_semantic_evidence(candidate_list, history_items if history_items is not None
                                    else self.dataset.get_user_train_items(user_id))
        item_mems = self._get_item_mems(candidates)
        if self.perturb:
            item_mems = self._perturb(user_id, candidate_list, item_mems)
        
        # Call different methods based on reranker mode
        if self.reranker_mode == "llm":
            recent_titles = None
            if self.recent_k > 0:
                hist = history_items if history_items is not None else self.dataset.get_user_train_items(user_id)
                recent_titles = [self._item_text(i, 160) for i in hist[-self.recent_k:]]
            rerank_scores = self.reranker.rerank(
                user_id=user_id,
                retrieval_bundle=retrieval_bundle,
                candidates=candidate_list,
                item_mems=item_mems,
                instruction=instruction,  # Pass instruction to LLM reranker
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                debug_logger=debug_logger,
                history_titles=recent_titles
            )
        else:  # vector mode
            rerank_scores = self.reranker.rerank(
                user_id=user_id,
                retrieval_bundle=retrieval_bundle,
                candidates=candidate_list,
                item_mems=item_mems,
                debug_logger=debug_logger
            )
        stage_rr_time = time.time() - t0
        self.n_stage_rr_calls += 1
        
        if self.debug:
            print(f"  Stage-ReRank ({self.reranker_mode}): {len(rerank_scores)} scores, {stage_rr_time*1000:.1f}ms")
        
        # 5. Sort (descending) - Critical: must sort by score!
        rerank_scores_sorted = sorted(rerank_scores, key=lambda x: x.get('score', 0), reverse=True)
        ranked_items = [s['item_id'] for s in rerank_scores_sorted]
        scores = [s['score'] for s in rerank_scores_sorted]
        
        # Build detailed information
        details = {
            'facets': retrieval_bundle.get('facets', []),
            'support_edges': retrieval_bundle.get('support_edges', []),
            'rerank_scores': rerank_scores,
            'scores': scores,
            'retrieval_bundle': retrieval_bundle,
            'pruned_subgraph': pruned_subgraph,
            'packed_context': packed_context,
            'ttm': None if ttm_state is None else {
                'trajectory_ids': [n['id'] for n in pruned_subgraph['neighbors'] if n.get('trajectory')],
                'candidate_scores': {int(c): float(ttm_state['vec'][c]) for c in candidates}},
            'timings': {
                'pruning': pruning_time,
                'stage_r': stage_r_time,
                'stage_rr': stage_rr_time
            }
        }
        
        if return_details:
            return ranked_items, details
        else:
            return ranked_items, {}
    
    def write(
        self,
        user_id: int,
        feedback: Dict,
        recent_facets: List[Dict],
        pruned_subgraph: Dict = None,
        debug_logger = None
    ) -> Dict:
        """Execute Stage-W: plan (LLM call on current memory) then apply immediately"""
        plan = self.plan_write(user_id, feedback, recent_facets, pruned_subgraph, debug_logger)
        return self.apply_write(plan, debug_logger)
    
    def plan_write(
        self,
        user_id: int,
        feedback: Dict,
        recent_facets: List[Dict],
        pruned_subgraph: Dict = None,
        debug_logger = None
    ) -> Dict:
        """
        Execute Stage-W (Write)
        
        Args:
            user_id: User ID
            feedback: Feedback information, e.g., {'action': 'CLICK', 'item_id': 123}
            recent_facets: Recent facets (from Stage-R)
            pruned_subgraph: Pruned subgraph (optional, for obtaining neighbor IDs)
            debug_logger: Optional debug logger
            
        Returns:
            {
                'user_patches': [...],
                'neighbor_patches': [...],
                'propagation_policy': {...},
                'stats': {...}
            }
        """
        if self.isolated:
            pruned_subgraph = None
        # Prepare current user memory
        current_profile = self.storage.get_user_memory(user_id)
        user_mem_keys = [current_profile] if current_profile else []
        
        # Prepare clicked item's current memory
        item_mem_keys = {}
        clicked_item_info = None
        if 'item_id' in feedback:
            item_id = feedback['item_id']
            current_item_mem = self.storage.get_item_memory(item_id)
            item_mem_keys[item_id] = [current_item_mem] if current_item_mem else []
            
            # Get item metadata (title) for generating semantic memory
            if self.dataset.item_metadata:
                item_text = self.dataset.item_metadata.get(item_id, {}).get('title', f'Item-{item_id}')
                clicked_item_info = f"[Item-{item_id}] {item_text}"
            else:
                clicked_item_info = f"[Item-{item_id}]"
        
        # Extract neighbor IDs (for propagation)
        neighbor_ids = []
        neighbor_map = {}  # id_string -> (nb_id, nb_type)
        neighbor_details = []  # Neighbor detailed information
        
        # Extract neighbors used in retrieval
        used_neighbors = set()
        for facet in recent_facets:
            supporting = facet.get('supporting_neighbors', [])
            used_neighbors.update(supporting)
        
        if pruned_subgraph:
            for nb in pruned_subgraph.get('neighbors', []):
                nb_id = nb['id']
                nb_type = nb['type']
                # Format as "User-123" or "Item-456"
                id_str = f"{nb_type.capitalize()}-{nb_id}"
                neighbor_ids.append(id_str)
                neighbor_map[id_str] = (nb_id, nb_type)
                
                # Collect neighbor detailed information
                nb_memory = ""
                if nb_type == 'user':
                    nb_memory = self.storage.get_user_memory(nb_id)
                else:
                    nb_memory = self.storage.get_item_memory(nb_id)
                
                # Clean memory display (filter Nan, empty strings, etc.)
                if nb_memory:
                    # If list (old format), convert to string
                    if isinstance(nb_memory, list):
                        nb_memory = str(nb_memory)
                    # Filter out 'Nan', 'nan', empty strings
                    if nb_memory.lower() in ['nan', '[nan]', "['']", '']:
                        nb_memory = "(No memory)"
                else:
                    nb_memory = "(No memory)"
                
                neighbor_details.append({
                    'id_str': id_str,
                    'score': nb.get('score', 0.0),
                    'memory': nb_memory,
                    'used_in_retrieval': id_str in used_neighbors
                })
        
        # Call Stage-W
        stage_w_response = self.manager.run_stage_w(
            user_id=user_id,
            feedback=feedback,
            recent_facets=recent_facets,
            user_mem_keys=user_mem_keys,
            item_mem_keys=item_mem_keys,
            neighbor_ids=neighbor_ids,
            neighbor_details=neighbor_details,  # New
            fanout_cap=self.fanout_cap,
            clicked_item_info=clicked_item_info,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            debug_logger=debug_logger
        )
        self.n_stage_w_calls += 1
        
        return {
            'user_id': user_id,
            'feedback': feedback,
            'stage_w_response': stage_w_response,
            'user_mem_keys': user_mem_keys,
            'item_mem_keys': item_mem_keys,
            'neighbor_map': neighbor_map,
            'neighbor_details': neighbor_details,
        }
    
    def apply_write(self, plan: Dict, debug_logger = None) -> Dict:
        """Apply a Stage-W plan to memory storage (user, clicked item, propagated neighbors)"""
        user_id = plan['user_id']
        feedback = plan['feedback']
        stage_w_response = plan['stage_w_response']
        user_mem_keys = plan['user_mem_keys']
        item_mem_keys = plan['item_mem_keys']
        neighbor_map = plan['neighbor_map']
        neighbor_details = plan['neighbor_details']
        
        # Get updated memories
        new_user_mem = stage_w_response.get('user_memory', '')
        new_item_mem = stage_w_response.get('item_memory', '')
        neighbor_updates = stage_w_response.get('neighbor_updates', [])
        
        # Output comparison of original and updated memories (for debugging)
        if debug_logger:
            debug_logger("\n>>> MEMORY UPDATE COMPARISON <<<")
            
            # User memory comparison
            old_user_mem = user_mem_keys[0] if user_mem_keys else "(No memory yet)"
            debug_logger(f"\n[User Memory]")
            debug_logger(f"  OLD: {old_user_mem}")
            debug_logger(f"  NEW: {new_user_mem}")
            
            # Item memory comparison
            old_item_mem = None
            if item_mem_keys:
                for item_id, mem_list in item_mem_keys.items():
                    if mem_list:
                        old_item_mem = mem_list[0]
                        break
            old_item_mem = old_item_mem if old_item_mem else "(No memory yet)"
            debug_logger(f"\n[Item Memory]")
            debug_logger(f"  OLD: {old_item_mem}")
            debug_logger(f"  NEW: {new_item_mem}")
            
            # Neighbor memory comparison
            if neighbor_updates:
                debug_logger(f"\n[Neighbor Memory Updates]")
                for update in neighbor_updates:
                    neighbor_id_str = update.get('neighbor_id', '')
                    memory_update = update.get('memory_update', '')
                    rationale = update.get('rationale', '')
                    
                    # Find original memory
                    old_neighbor_mem = "(No memory)"
                    if neighbor_details:
                        nb_detail = next((d for d in neighbor_details if d.get('id_str') == neighbor_id_str), None)
                        if nb_detail:
                            old_neighbor_mem = nb_detail.get('memory', '(No memory)')
                    
                    debug_logger(f"\n  [{neighbor_id_str}]")
                    debug_logger(f"    OLD: {old_neighbor_mem}")
                    debug_logger(f"    NEW: {memory_update}")
                    debug_logger(f"    Rationale: {rationale}")
            else:
                debug_logger(f"\n[Neighbor Memory Updates] (None)")
            
            debug_logger("")  # Empty line separator
        
        # Update user memory
        if new_user_mem:
            self.storage.update_user_memory(user_id, new_user_mem)
            user_updated = 1
        else:
            user_updated = 0
        
        # Update item memory
        clicked_item_id = feedback.get('item_id')
        if new_item_mem and clicked_item_id:
            self.storage.update_item_memory(clicked_item_id, new_item_mem)
            item_updated = 1
        else:
            item_updated = 0
        
        # Neighbor propagation (decided by LLM)
        neighbor_updated = 0
        user_neighbors_propagated = []
        item_neighbors_propagated = []
        for update in neighbor_updates:
            neighbor_id_str = update.get('neighbor_id', '')
            memory_update = update.get('memory_update', '')
            rationale = update.get('rationale', '')
            
            if not neighbor_id_str or not memory_update:
                continue
            
            # Parse neighbor ID
            if neighbor_id_str not in neighbor_map:
                if debug_logger:
                    debug_logger(f"  ⚠️ Unknown neighbor: {neighbor_id_str}")
                continue
            
            nb_id, nb_type = neighbor_map[neighbor_id_str]
            
            # Update neighbor memory
            if nb_type == 'user':
                self.storage.update_user_memory(nb_id, memory_update)
                user_neighbors_propagated.append(nb_id)
                if debug_logger:
                    debug_logger(f"  ✓ Updated User-{nb_id}: {rationale}")
            else:  # item
                self.storage.update_item_memory(nb_id, memory_update)
                item_neighbors_propagated.append(nb_id)
                if debug_logger:
                    debug_logger(f"  ✓ Updated Item-{nb_id}: {rationale}")
            
            neighbor_updated += 1
        
        # Statistics
        combined_stats = {
            'user_applied': user_updated,
            'item_applied': item_updated,
            'neighbor_applied': neighbor_updated,
            'user_neighbors_propagated': user_neighbors_propagated,
            'item_neighbors_propagated': item_neighbors_propagated,
            'total_applied': user_updated + item_updated + neighbor_updated
        }
        
        if debug_logger:
            if neighbor_updated > 0:
                debug_logger(f"\n🔄 Collaborative Propagation (LLM-decided):")
                debug_logger(f"  User neighbors: {len(user_neighbors_propagated)} updated")
                debug_logger(f"  Item neighbors: {len(item_neighbors_propagated)} updated")
                debug_logger(f"  Total propagated: {neighbor_updated}")
            else:
                debug_logger(f"\n⚠️ No neighbor propagation (LLM chose not to propagate)")
        
        return {
            'user_memory': new_user_mem,
            'item_memory': new_item_mem,
            'stats': combined_stats,
            'raw_response': stage_w_response
        }
    
    def _rerank_vanilla(self, user_id, candidates, instruction, history_items, debug_logger):
        """Vanilla LLM baseline: one rerank call, no pruning/Stage-R/memory"""
        t0 = time.time()
        history_items = history_items if history_items is not None else self.dataset.get_user_train_items(user_id)
        recent = history_items[-self.vanilla_history_len:]
        history_titles = [self._item_text(i, 160) for i in recent]
        candidate_list = []
        for cid in candidates:
            meta = self.dataset.item_metadata.get(cid, {})
            candidate_list.append({'id': cid, 'title': meta.get('title', 'N/A'),
                                   'description': self._clean_desc(meta.get('description', '')), 'tags': []})
        ttm_state = None
        if self.ttm is not None and self.ttm_evidence:  # control: next-step evidence without MemRec memory
            ttm_state = self.ttm.user_state(history_items)
            evidence = self.ttm.candidate_evidence(ttm_state, candidates)
            for c in candidate_list:
                c['evidence'] = evidence[c['id']]
        self._add_semantic_evidence(candidate_list, history_items)
        rerank_scores = self.reranker.rerank(
            user_id=user_id, retrieval_bundle={}, candidates=candidate_list, item_mems={},
            instruction=instruction, temperature=self.temperature, max_tokens=self.max_tokens,
            debug_logger=debug_logger, vanilla_mode=True, history_titles=history_titles
        )
        self.n_stage_rr_calls += 1
        rerank_scores_sorted = sorted(rerank_scores, key=lambda x: x.get('score', 0), reverse=True)
        details = {'facets': [], 'rerank_scores': rerank_scores, 'pruned_subgraph': None,
                   'ttm': None if ttm_state is None else {
                       'candidate_scores': {int(c): float(ttm_state['vec'][c]) for c in candidates}},
                   'timings': {'pruning': 0.0, 'stage_r': 0.0, 'stage_rr': time.time() - t0}}
        return [x['item_id'] for x in rerank_scores_sorted], details
    
    def _add_semantic_evidence(self, candidate_list: List[Dict], history: List[int]):
        if self.trend is not None:
            trend = self.trend.candidate_evidence([c['id'] for c in candidate_list])
            for c in candidate_list:
                c['trend_evidence'] = trend[c['id']]
        if self.sem is None or not history:
            return
        evidence = self.sem.candidate_evidence(self.sem.user_state(history), [c['id'] for c in candidate_list])
        for c in candidate_list:
            c['sem_evidence'] = evidence[c['id']]

    @staticmethod
    def _clean_desc(desc) -> str:
        desc = str(desc) if desc is not None else ''
        return '' if desc.strip().lower() in ('', 'nan', '[]', "['']", 'none') else desc.strip()
    
    def _item_text(self, item_id: int, max_len: int) -> str:
        meta = self.dataset.item_metadata.get(item_id, {}) if self.dataset.item_metadata else {}
        title = meta.get('title', f'Item-{item_id}')
        desc = self._clean_desc(meta.get('description', ''))
        text = f"{title}. {desc}" if desc else title
        return text[:max_len]
    
    def _timeline_text(self, user_id: int, history: List[int]) -> str:
        """B1: last timeline_k items of the strict-past history, oldest first, with time before the latest one"""
        times = self.dataset.user_times.get(user_id, [])[:len(history)]
        recent = list(zip(history, times))[-self.timeline_k:]
        t_last = recent[-1][1] if recent else 0
        lines = []
        for n, (item, t) in enumerate(recent, 1):
            days = (t_last - t) / 86400.0
            when = 'same day as the latest' if days < 1 else f'{days:.0f} days before the latest'
            lines.append(f"  {n}. {self._item_text(item, 120)} ({when})")
        return "\n".join(lines)

    def _prepare_candidate_list(self, candidates: List[int]) -> List[Dict]:
        """
        Prepare candidate list (add metadata)
        
        Args:
            candidates: List of candidate item IDs
            
        Returns:
            [{'id': int, 'title': str, 'tags': [...], ...}, ...]
        """
        candidate_list = []
        for cid in candidates:
            item_meta = self.dataset.item_metadata.get(cid, {})
            candidate_list.append({
                'id': cid,
                'title': item_meta.get('title', 'N/A'),
                'tags': item_meta.get('tags', [])
            })
        return candidate_list
    
    def _perturb(self, user_id, candidate_list, item_mems):
        """Evaluation-time memory ablation (runner sets self.perturb after warm-up): derange the candidates' item
        memories ('item_mem') and/or transition evidence ('evidence') among the candidates, seeded by user. The prompt
        keeps its length and format, but each memory now describes another candidate"""
        import random
        rng = random.Random(1000003 * int(user_id) + 17)
        ids = [c['id'] for c in candidate_list]
        perm = ids[:]
        while len(ids) > 1 and any(a == b for a, b in zip(ids, perm)):
            rng.shuffle(perm)
        src = dict(zip(ids, perm))  # candidate i shows the memory of candidate src[i]
        if 'item_mem' in self.perturb:
            item_mems = {i: item_mems[src[i]] for i in ids if src[i] in item_mems}
        if 'evidence' in self.perturb:
            ev = {c['id']: c.get('evidence') for c in candidate_list}
            for c in candidate_list:
                if ev.get(src[c['id']]) is not None:
                    c['evidence'] = ev[src[c['id']]]
        return item_mems

    def _get_item_mems(self, candidates: List[int]) -> Dict[int, Dict]:
        """
        Get memories of candidate items
        
        Args:
            candidates: List of candidate item IDs
            
        Returns:
            {item_id: ItemMem, ...}
        """
        item_mems = {}
        for cid in candidates:
            item_mem = self.storage.get_item_memory(cid)
            if item_mem:
                item_mems[cid] = item_mem
        return item_mems
    
    def _extract_scores(
        self, 
        candidates: List[int], 
        candidate_notes: List[Dict]
    ) -> List[float]:
        """
        Extract scores from candidate_notes (deprecated, v2 uses reranker)
        
        Args:
            candidates: Candidate list
            candidate_notes: LM-generated candidate_notes
            
        Returns:
            List of scores corresponding to candidates
        """
        # Build mapping
        note_map = {note['candidate_id']: note['fit'] for note in candidate_notes}
        
        # Extract scores (0.0 if not available)
        scores = [note_map.get(cid, 0.0) for cid in candidates]
        
        return scores
    
    def get_stats(self) -> Dict:
        """Get statistics"""
        storage_stats = self.storage.get_stats()
        
        return {
            'n_stage_r_calls': self.n_stage_r_calls,
            'n_stage_rr_calls': self.n_stage_rr_calls,
            'n_stage_w_calls': self.n_stage_w_calls,
            'reranker_mode': self.reranker_mode,
            'pruner_mode': self.pruner_mode,
            'storage': storage_stats,
            'graph': self.graph.get_stats()
        }
