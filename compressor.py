import torch
import numpy as np
from typing import List, Optional, Dict, Union, Tuple
from torch import Tensor
import time
import re
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

class PromptCompressor:
    """
    A comprehensive prompt compressor that supports compression 
    using entropy and Attention scores with additive/multiplicative fusion.
    """

    def __init__(self, model_name, device_map: str = "cuda" if torch.cuda.is_available() else "cpu", model_config: dict = {},):
        """
        Initialize the compressor with model and tokenizer.
        
        Args:
            model_name: Pretrained language model.
            device_map: Device to run compression model.
        """
        self.model_name = model_name
        self.load_model(model_name, device_map, model_config)
        

    def load_model(self, model_name, device_map: str = "cuda", model_config: dict = {}):
        
        trust_remote_code = model_config.get("trust_remote_code", True)
        if "trust_remote_code" not in model_config:
            model_config["trust_remote_code"] = trust_remote_code
        config = AutoConfig.from_pretrained(model_name, **model_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name, **model_config)
        if model_config.get("pad_to_left", True):
            tokenizer.padding_side = "left"
            tokenizer.pad_token_id = (
                config.pad_token_id if config.pad_token_id else tokenizer.eos_token_id
            )
        MODEL_CLASS = (
                AutoModelForTokenClassification
                if any("ForTokenClassification" in ar for ar in config.architectures)
                else AutoModelForCausalLM
        )
        self.device = (
                device_map
                if any(key in device_map for key in ["cuda", "cpu", "mps"])
                else "cuda"
        )
        model = MODEL_CLASS.from_pretrained(
                model_name,
                torch_dtype=model_config.pop(
                    "torch_dtype", "auto" if device_map == "cuda" else torch.float32
                ),
                device_map=device_map,
                config=config,
                ignore_mismatched_sizes=True,
                attn_implementation="eager",
                **model_config,
                )
        self.tokenizer = tokenizer
        self.model = model
        self.context_idxs = []
        self.max_position_embeddings = config.max_position_embeddings
        self.model_config = config

    def normalize(self, tensor: Tensor) -> Tensor:
        """
        Min-max normalize tensor to [0, 1].
        """
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val - min_val > 1e-8:
            return (tensor - min_val) / (max_val - min_val)
        else:
            return torch.zeros_like(tensor)

    def get_ppl(self, context: str = "", input_ids: Tensor = None, attention_mask: Tensor = None, return_attn: bool = False) -> Tuple:
        """
        Compute perplexity (PPL) for each token. Optionally return attention scores.
        
        Args:
            context: Input text.
            input_ids: Optional pre-encoded input IDs.
            attention_mask: Optional attention mask.
            return_attn: Whether to return attention sum.
        
        Returns:
            Tuple of (ppl, input_ids, attention_mask, [attn_sum])
        """
        with torch.no_grad():
            if input_ids is None:
                inputs = self.tokenizer(context, return_tensors="pt", add_special_tokens=False)
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)
            else:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

            self.model.eval()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=return_attn,
                return_dict=True
            )
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            token_losses = token_losses.view(shift_labels.size())

            if return_attn:
                all_heads_sum = None
                matrix_num = 0
                for layer in range(len(outputs.attentions)):
                    for head in range(outputs.attentions[layer].shape[1]):
                        if all_heads_sum is None:
                            all_heads_sum = outputs.attentions[layer][0][head].squeeze()
                        else:
                            matrix_num += 1
                            all_heads_sum += outputs.attentions[layer][0][head].squeeze()
                column_sum = torch.sum(all_heads_sum, dim=0)[1:] / matrix_num
                return token_losses, input_ids, attention_mask, column_sum
            else:
                return token_losses, input_ids, attention_mask

    def _fuse_attn_ppl_additive(self, ppl: Tensor, attn: Tensor, alpha: float = 0.8) -> Tensor:
        """
        Fuse PPL and Attention using additive rule: score = alpha * attn + (1-alpha) * ppl
        
        Args:
            ppl: Perplexity scores.
            attn: Attention scores.
            alpha: Weight for attention.
        Returns:
            Fused score.
        """
        ppl_norm = self.normalize(ppl)
        attn_norm = self.normalize(attn)
    
        score = alpha * attn_norm + (1 - alpha) * ppl_norm
        return score

    def _fuse_attn_ppl_multiplicative(self, ppl: Tensor, attn: Tensor, eps: float = 1e-8) -> Tensor:
        """
        Fuse PPL and Attention using multiplicative rule: score = attn * (1/ppl)
        
        Args:
            ppl: Perplexity scores.
            attn: Attention scores.
        Returns:
            Fused score.
        """
        score = new_ppl = torch.mul(ppl, attn)
        return score

    def _preserve_punctuation_mask(self, input_ids: Tensor, device: str) -> Tensor:
        """
        Return a boolean mask indicating which tokens are punctuation/special and should be preserved.
        """
        ids = input_ids[0].cpu().numpy()
        decoded_tokens = [self.tokenizer.decode([id_]) for id_ in ids]
        preserve = []
        punct_pattern = re.compile(r'^\s*[^\w\s]+\s*$')
        for token in decoded_tokens:
            is_punct = bool(punct_pattern.match(token))
            is_special = token in ["<s>", "</s>", "[CLS]", "[SEP]", "<pad>"]
            preserve.append(is_punct or is_special)
        return torch.tensor(preserve, dtype=torch.bool, device=device)

    def direct_compress(
        self,
        ppl: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        compress_ratio: float,
        preserve_punct: bool = True
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compress by keeping top (1-ratio) tokens with highest scores.
        Optionally preserve punctuation.
        """
        if compress_ratio <= 0:
            return input_ids, attention_mask, torch.arange(input_ids.size(1))

        total_tokens = ppl.numel()
        k = int(total_tokens * (1 - compress_ratio))
        k = max(1, k)

        if preserve_punct:
            punct_mask = self._preserve_punctuation_mask(input_ids, ppl.device)
            ppl = ppl + 1e5 * punct_mask

        _, indices = torch.topk(ppl.view(-1), k=k, largest=True)
        sorted_indices = torch.sort(indices)[0]

        selected_input_ids = input_ids[:, sorted_indices]
        selected_attention_mask = attention_mask[:, sorted_indices]

        return selected_input_ids, selected_attention_mask, sorted_indices

    def direct_compress_attn(
        self,
        ppl: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        compress_ratio: float,
        attn_sum: Tensor,
        fusion: str = "additive",
        alpha: float = 0.8,
        preserve_punct: bool = True
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compress using fused PPL and Attention scores.
        """
        if compress_ratio <= 0:
            return input_ids, attention_mask, torch.arange(input_ids.size(1))

        if fusion == "additive":
            score = self._fuse_attn_ppl_additive(ppl, attn_sum, alpha)
        elif fusion == "multiplicative":
            score = self._fuse_attn_ppl_multiplicative(ppl, attn_sum)
        else:
            raise ValueError("Fusion must be 'additive' or 'multiplicative'")

        total_tokens = score.numel()
        k = int(total_tokens * (1 - compress_ratio))
        k = max(1, k)

        if preserve_punct:
            punct_mask = self._preserve_punctuation_mask(input_ids, score.device)
            score = score + 1e5 * punct_mask

        _, indices = torch.topk(score.view(-1), k=k, largest=True)
        sorted_indices = torch.sort(indices)[0]

        selected_input_ids = input_ids[:, sorted_indices]
        selected_attention_mask = attention_mask[:, sorted_indices]
        new_attn_sum = attn_sum[sorted_indices]

        return selected_input_ids, selected_attention_mask, sorted_indices, new_attn_sum

    def direct_compress_attn_wosucce(
        self,
        ppl: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        compress_ratio: float,
        attn_sum: Tensor,
        fusion: str = "additive",
        alpha: float = 0.8,
        preserve_punct: bool = True
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compress using fused PPL and Attention scores.
        """
        if compress_ratio <= 0:
            return input_ids, attention_mask, torch.arange(input_ids.size(1))

        if fusion == "additive":
            score = self._fuse_attn_ppl_additive(ppl, attn_sum, alpha)
        elif fusion == "multiplicative":
            score = self._fuse_attn_ppl_multiplicative(ppl, attn_sum)
        else:
            raise ValueError("Fusion must be 'additive' or 'multiplicative'")

        total_tokens = score.numel()
        k = int(total_tokens * (1 - compress_ratio))
        k = max(1, k)

        if preserve_punct:
            punct_mask = self._preserve_punctuation_mask(input_ids, score.device)
            score = score - 1e5 * punct_mask

        _, indices = torch.topk(score.view(-1), k=k, largest=True)
        sorted_indices = torch.sort(indices)[0]

        all_values = torch.arange(ppl.numel()).to(self.device)
        del_indices = all_values[~torch.isin(all_values, sorted_indices)]
        differences = del_indices[1:] - del_indices[:-1]
        mask = torch.ones_like(del_indices, dtype=torch.bool)
        mask[1:] = differences == 1
        mask[0] = False
        
        for i in range(1, len(mask)):
            if mask[i-1]:
                mask[i] = False
        filtered_indices = del_indices[mask]
        all_indices, _ = torch.sort(torch.cat((indices, filtered_indices)))
        
        selected_input_ids = input_ids[:, all_indices]
        selected_attention_mask = attention_mask[:, all_indices]
        new_attn_sum = attn_sum[all_indices]

        return selected_input_ids, selected_attention_mask, sorted_indices, new_attn_sum
    
    def get_decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to string."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def compress(
        self,
        context: str,
        compress_ratio: float = 0.5,
        method: str = "attn_ppl",
        fusion: str = "additive",
        alpha: float = 0.8,
        dyn_time: Optional[int] = None,
        preserve_punct: bool = False,
        return_info: bool = True
    ) -> Union[str, Dict[str, any]]:
        """
        Compression interface supporting multiple strategies.
        
        Args:
            context: Input text.
            compress_ratio: Compression ratio (0 ~ 1).
            method: "ppl", "attn_ppl", "dynamic_ppl", "dynamic_attn_ppl", and "dynamic_attn_ppl_wosucce"
            fusion: "additive" or "multiplicative".
            alpha: Weight for attention in additive fusion.
            dyn_time: Number of dynamic iterations. If None, auto-calculate.
            preserve_punct: Whether to preserve punctuation and special tokens.
            return_info: If True, return dict with details; else return string.
        
        """
        start_time = time.time()

        if not context.strip():
            context = " "

        assert 0 <= compress_ratio < 1, "compress_ratio must be in [0, 1)"

        ppl, input_ids, attention_mask = self.get_ppl(context)
        seq_len = input_ids.size(1)
        
        if dyn_time == None:
            dyn_time = min(max(1, seq_len // 100), 15)

        if method == "ppl":
            ppl, input_ids, attention_mask = self.get_ppl(context)
            selected_input_ids, selected_attention_mask, kept_indices = self.direct_compress(
                ppl, input_ids[:, 1:], attention_mask[:, 1:], compress_ratio, preserve_punct
            )
            selected_input_ids = torch.cat((input_ids[:, :1], selected_input_ids), dim=1)
            selected_attention_mask = torch.cat((attention_mask[:, :1], selected_attention_mask), dim=1)

        elif method == "attn_ppl":
            ppl, input_ids, attention_mask, attn_sum = self.get_ppl(context, return_attn=True)
            selected_input_ids, selected_attention_mask, kept_indices, _ = self.direct_compress_attn(
                ppl, input_ids[:, 1:], attention_mask[:, 1:], compress_ratio, attn_sum,
                fusion=fusion, alpha=alpha, preserve_punct=preserve_punct
            )
            selected_input_ids = torch.cat((input_ids[:, :1], selected_input_ids), dim=1)
            selected_attention_mask = torch.cat((attention_mask[:, :1], selected_attention_mask), dim=1)

        elif method == "dynamic_ppl":
            ppl, input_ids, attention_mask = self.get_ppl(context)
            real_ratio = (1 - (1 - compress_ratio) ** (1.0 / dyn_time))
            for _ in range(dyn_time):
                selected_input_ids, selected_attention_mask, kept_indices = self.direct_compress(
                    ppl, input_ids[:, 1:], attention_mask[:, 1:], real_ratio, preserve_punct
                )
                selected_input_ids = torch.cat((input_ids[:, :1], selected_input_ids), dim=1)
                selected_attention_mask = torch.cat((attention_mask[:, :1], selected_attention_mask), dim=1)
                ppl, input_ids, attention_mask = self.get_ppl("", input_ids=selected_input_ids, attention_mask=selected_attention_mask)

        elif method == "dynamic_attn_ppl":
            ppl, input_ids, attention_mask, attn_sum = self.get_ppl(context, return_attn=True)
            real_ratio = (1 - (1 - compress_ratio) ** (1.0 / dyn_time))
            for _ in range(dyn_time):
                selected_input_ids, selected_attention_mask, kept_indices, attn_sum = self.direct_compress_attn(
                    ppl, input_ids[:, 1:], attention_mask[:, 1:], real_ratio, attn_sum,
                    fusion=fusion, alpha=alpha, preserve_punct=preserve_punct
                )
                selected_input_ids = torch.cat((input_ids[:, :1], selected_input_ids), dim=1)
                selected_attention_mask = torch.cat((attention_mask[:, :1], selected_attention_mask), dim=1)
                ppl, input_ids, attention_mask, attn_sum = self.get_ppl("", input_ids=selected_input_ids, attention_mask=selected_attention_mask, return_attn=True)

        elif method == "dynamic_attn_ppl_wosucce":
            ppl, input_ids, attention_mask, attn_sum = self.get_ppl(context, return_attn=True)
            real_ratio = (1 - (1 - compress_ratio) ** (1.0 / dyn_time))
            for _ in range(dyn_time):
                selected_input_ids, selected_attention_mask, kept_indices, attn_sum = self.direct_compress_attn_wosucce(
                    ppl, input_ids[:, 1:], attention_mask[:, 1:], real_ratio, attn_sum,
                    fusion=fusion, alpha=alpha, preserve_punct=preserve_punct
                )
                selected_input_ids = torch.cat((input_ids[:, :1], selected_input_ids), dim=1)
                selected_attention_mask = torch.cat((attention_mask[:, :1], selected_attention_mask), dim=1)
                ppl, input_ids, attention_mask, attn_sum = self.get_ppl("", input_ids=selected_input_ids, attention_mask=selected_attention_mask, return_attn=True)
                
        else:
            raise ValueError(f"Unknown method: {method}")

        decoded_text = self.get_decode(selected_input_ids[0].tolist())
        compressed_tokens = selected_input_ids.size(1)
        actual_ratio = 1 - (compressed_tokens / seq_len) if seq_len > 0 else 0

        result = {
            "compressed_text": decoded_text,
            "original_tokens": seq_len,
            "compressed_tokens": compressed_tokens,
            "actual_ratio": round(actual_ratio, 3),
            "kept_indices": kept_indices.tolist(),
            "compress_ratio": compress_ratio,
            "method": method,
            "fusion": fusion,
            "alpha": alpha,
            "dyn_time": dyn_time,
            "processing_time": round(time.time() - start_time, 3),
        }

        if return_info:
            return result
        else:
            return decoded_text