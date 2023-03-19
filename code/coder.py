import hashlib
import hmac
import logging
import math
import os
from dataclasses import dataclass, astuple
from typing import List, Dict, Tuple, Literal, Optional

import bitarray
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from trie import TokenTrie

sample_seed_prefix = b'sample'


def limit_past(past: Tuple[Tuple[torch.FloatTensor]]) -> List[Tuple[torch.FloatTensor]]:
    past = list(past)
    for i in range(len(past)):
        past[i] = list(past[i])
        for j in range(len(past[i])):
            past[i][j] = past[i][j][:, :, -1022:, :]
        past[i] = tuple(past[i])
    return past


def kl(q, logq, logp):
    res = q * (logq - logp) / 0.69315
    res[q == 0] = 0
    return res.sum().item()  # in bits


def entropy(q, logq):
    res = q * logq / 0.69315
    res[q == 0] = 0
    return -res.sum().item()  # in bits


# e.g. [0, 1, 1, 1] looks like 1110=14
def bits2int(bits):
    res = 0
    for i, bit in enumerate(bits):
        res += bit * (2 ** i)
    return res


def int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}' % num_bits).format(inp)
    return [int(strval) for strval in reversed(strlist)]


def is_sent_finish(token_idx, enc):
    token = enc.decoder[token_idx]
    return '.' in token or '!' in token or '?' in token


def num_same_from_beg(bits1, bits2):
    assert len(bits1) == len(bits2)
    assert len(bits1) > 0
    for i in range(len(bits1)):
        if bits1[i] != bits2[i]:
            break

    return i


def encode_context(raw_text, enc) -> List[int]:
    context_tokens = [enc.encoder['<|endoftext|>']] + enc.encode(raw_text, truncation=True)
    return context_tokens


def bin_sort(l, token_indices, total, entropy, device):
    # compute entropy for upper bound on the number of bins we need

    bucket_size = total
    num_bins = 2 ** int(entropy + 1)
    bucket_size = total / num_bins

    bins = [torch.empty(0, dtype=torch.long, device=device)] * num_bins
    value_in_bins = [0] * num_bins
    space_left_after = [total - i * bucket_size for i in range(0, num_bins)]

    token_bins = [torch.empty(0, dtype=torch.long, device=device)] * num_bins

    # Figuring out what the search order should be
    step_size = num_bins / 4
    search_order = []
    priorities = [0] * num_bins
    priority = 0
    search_order.append(int(num_bins / 2))
    search_order.append(0)
    priorities[int(num_bins / 2)] = 0
    priorities[0] = 0
    while (step_size >= 1):
        priority += 1
        for x in range(num_bins - int(step_size), -1, -int(step_size * 2)):
            search_order.append(x)
            priorities[x] = priority
        step_size = step_size / 2

    # Adding the actual elements
    for (item, token_index) in zip(l.tolist(), token_indices.tolist()):
        found_single_bucket_fit = False
        single_bucket_index = -1
        single_bucket_value = bucket_size

        found_multi_bucket_bumpless_fit = False
        multi_bucket_bumpless_index = -1
        multi_bucket_bumpless_value = total

        found_multi_bucket_bumping_fit = False
        multi_bucket_bumping_index = -1
        multi_bucket_bumping_value = total

        for i in search_order:  # for index in search_order
            if (item > space_left_after[i]):
                continue
            if (value_in_bins[i] >= bucket_size):
                continue

            # Priority of choices
            #  1. Can i place this thing in an empty bucket all on its own?
            #  2. Can i plan this somewhere where is doesnt have to bump anything else around?
            #    2a. Minimize the wasted space.  Aka use the smallest space (of equal priority) that accomplishes this goal
            #  3. If not (1) and (2), then put it in the space the bumps stuff the least.

            if (value_in_bins[i] + item > bucket_size):  # Would overflow.

                space_before_next_block = bucket_size - value_in_bins[i]
                for j in range(i + 1, len(bins)):
                    if (value_in_bins[
                        j] > 0):  # We have found a bucket with something in it.  This is how much space we have here.
                        space_before_next_block = space_before_next_block + (bucket_size - value_in_bins[i])
                        break
                    else:  # This was a empty bucket
                        space_before_next_block = space_before_next_block + bucket_size

                if ((not found_multi_bucket_bumpless_fit) or (
                        found_multi_bucket_bumpless_fit and priorities[i] <= priorities[
                    multi_bucket_bumpless_index])):  # This could potentially be a match

                    # If this is a valid space to put this without bumping and it is a better fit than previous spaces
                    if (space_before_next_block > item and space_before_next_block < multi_bucket_bumpless_value):
                        # set this to be the pointer!  we can fit stuff here
                        found_multi_bucket_bumpless_fit = True
                        multi_bucket_bumpless_index = i
                        multi_bucket_bumpless_value = space_before_next_block

                    # Find the overflow that will bump the least
                    if (item - space_before_next_block < multi_bucket_bumping_value):
                        found_multi_bucket_bumping_fit = True
                        multi_bucket_bumping_index = i
                        multi_bucket_bumping_value = item - space_before_next_block

            if (value_in_bins[i] + item <= bucket_size):  # Would fit
                if (single_bucket_value > value_in_bins[i]):
                    found_single_bucket_fit = True
                    single_bucket_value = value_in_bins[i]
                    single_bucket_index = i

        if (single_bucket_index == multi_bucket_bumpless_index == multi_bucket_bumping_index == -1):
            bins[0] = torch.cat((torch.tensor([item], device=device), bins[0]), 0)
            token_bins[0] = torch.cat((torch.tensor([token_index], device=device), token_bins[0]), 0)
            continue

        if found_single_bucket_fit:
            # We found somewhere we can actually fit!
            bins[single_bucket_index] = torch.cat((bins[single_bucket_index], torch.tensor([item], device=device)), 0)
            token_bins[single_bucket_index] = torch.cat(
                (token_bins[single_bucket_index], torch.tensor([token_index], device=device)), 0)
            value_in_bins[single_bucket_index] += item
            for i in range(0, single_bucket_index + 1):
                space_left_after[i] -= item

        elif found_multi_bucket_bumpless_fit:
            # Found somewhere we can put this without upsetting the force
            part_in_bucket = bucket_size - value_in_bins[multi_bucket_bumpless_index]
            part_overflow = item - part_in_bucket
            bins[multi_bucket_bumpless_index] = torch.cat(
                (bins[multi_bucket_bumpless_index], torch.tensor([item], device=device)), 0)
            token_bins[multi_bucket_bumpless_index] = torch.cat(
                (token_bins[multi_bucket_bumpless_index], torch.tensor([token_index], device=device)), 0)
            value_in_bins[multi_bucket_bumpless_index] = bucket_size

            # Fill this bucket and continue overflowing
            j = multi_bucket_bumpless_index + 1
            for i in range(0, j):
                space_left_after[i] -= item

            while (part_overflow > 0):
                new_part_overflow = (value_in_bins[j] + part_overflow) - bucket_size
                value_in_bins[j] = min(bucket_size, part_overflow + value_in_bins[j])  # mark the bucket as filled
                space_left_after[j] -= part_overflow
                part_overflow = new_part_overflow
                j += 1

        else:
            part_in_bucket = bucket_size - value_in_bins[multi_bucket_bumping_index]
            part_overflow = item - part_in_bucket
            bins[multi_bucket_bumping_index] = torch.cat(
                (bins[multi_bucket_bumping_index], torch.tensor([item], device=device)), 0)
            token_bins[multi_bucket_bumping_index] = torch.cat(
                (token_bins[multi_bucket_bumping_index], torch.tensor([token_index], device=device)), 0)
            value_in_bins[multi_bucket_bumping_index] = bucket_size

            # Fill this bucket and continue overflowing
            j = multi_bucket_bumping_index + 1
            for i in range(0, j):
                space_left_after[i] -= item
            while (part_overflow > 0):
                new_part_overflow = (value_in_bins[j] + part_overflow) - bucket_size
                value_in_bins[j] = min(bucket_size, part_overflow + value_in_bins[j])  # mark the bucket as filled
                space_left_after[j] -= part_overflow
                part_overflow = new_part_overflow
                j += 1

    sorted_tensor = torch.cat(bins, 0)
    sorted_tokens = torch.cat(token_bins, 0)

    return sorted_tensor, sorted_tokens


# TODO implement randomized
def encode_conversation_meteor(model, enc, message, context: List[int], key, nonce, finish_sent=True, device='cuda',
                               temp=1.0, precision=16, topk=50000, is_sort=False, randomized=False):
    mask_generator = DRBG(key, sample_seed_prefix + nonce)
    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)

    max_val = 2 ** precision
    threshold = 2 ** (-precision)
    cur_interval = [0, max_val]  # bottom inclusive, top exclusive

    prev = context
    output = context
    encoded_bits_in_output = []
    past = None

    total_num = 0
    total_num_for_stats = 0
    total_log_probs = 0
    total_kl = 0  # in bits
    total_entropy_ptau = 0

    with torch.no_grad():
        i = 0
        sent_finish = False
        while (i < len(message) or (finish_sent and not sent_finish)):
            # logging.debug(f'{i}: {i / float(len(message))}')
            input_ids, kwargs = model.prepare_model_inputs(prev)
            kwargs['past_key_values'] = past
            result = model(input_ids.unsqueeze(0), **kwargs)
            logits = result.logits
            past = result.past_key_values
            past = limit_past(past)
            # if i < 10:  # embed at least 10 bits in stegotext
            logits[0, -1, -1] = -1e20  # endoftext token can't happen
            logits[0, -1, 628] = -1e20  # 2 newlines token can't happen
            logits, indices = logits[0, -1, :].sort(descending=True)
            logits = logits.double()
            logits_temp = logits / temp
            probs_temp = F.softmax(logits_temp, dim=0)
            log_probs_temp = F.log_softmax(logits_temp, dim=0)
            log_probs = F.log_softmax(logits, dim=0)

            # conditions for having reached the end of the message
            if i >= len(message):
                selection = 0
                sent_finish = is_sent_finish(indices[selection].item(), enc)
            else:
                # Cutoff low probabilities that would be rounded to 0
                cur_int_range = cur_interval[1] - cur_interval[0]
                cur_threshold = 1 / cur_int_range
                k = min(max(2, (probs_temp < cur_threshold).nonzero()[0].item()), topk)
                probs_temp_int = probs_temp[:k]  # Cutoff all but top k
                old_indices = indices
                indices = indices[:k]

                # Rescale to correct range
                probs_temp_int = probs_temp_int / probs_temp_int.sum() * cur_int_range

                entropy_in_this_distribution = entropy(probs_temp, log_probs_temp)

                # Round probabilities to integers given precision
                probs_temp_int = probs_temp_int.round().long()

                if is_sort:
                    probs_temp_int, indices = bin_sort(probs_temp_int, indices, cur_int_range,
                                                       entropy_in_this_distribution, device)
                cum_probs = probs_temp_int.cumsum(0)

                # Remove any elements from the bottom if rounding caused the total prob to be too large
                overfill_index = (cum_probs > cur_int_range).nonzero()
                if len(overfill_index) > 0:
                    cum_probs = cum_probs[:overfill_index[0]]

                # Add any mass to the top if removing/rounding causes the total prob to be too small
                cum_probs += cur_int_range - cum_probs[-1]  # add

                # Get out resulting probabilities
                probs_final = cum_probs.clone()
                probs_final[1:] = cum_probs[1:] - cum_probs[:-1]

                # Convert to position in range
                cum_probs += cur_interval[0]

                # Apply the mask to the message
                message_bits = message[i:i + precision]
                if i + precision > len(message):
                    message_bits = message_bits + [0] * (i + precision - len(message))

                mask_bits = mask_generator.generate_bits(precision)

                for b in range(0, len(message_bits)):
                    message_bits[b] = message_bits[b] ^ mask_bits[b]

                # Get selected index based on binary fraction from message bits
                message_idx = bits2int(reversed(message_bits))
                selection = (cum_probs > message_idx).nonzero()[0].item()

                # Calculate new range as ints
                new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]
                new_int_top = cum_probs[selection]

                # Convert range to bits
                new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
                new_int_top_bits_inc = list(
                    reversed(int2bits(new_int_top - 1, precision)))  # -1 here because upper bound is exclusive

                # Consume most significant bits which are now fixed and update interval
                num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
                encoded_bits_in_output += [num_bits_encoded]  # for statistics
                i += num_bits_encoded

                # Gather statistics
                total_log_probs += log_probs[selection].item()

                q = probs_final.double() / probs_final.sum()
                logq = q.log()
                total_kl += kl(q, logq, log_probs[:len(q)])
                total_entropy_ptau += entropy_in_this_distribution
                total_num_for_stats += 1

            # Update history with new token
            prev = indices[selection].view(1)
            output = torch.cat((output, prev))
            total_num += 1

            if prev == enc.eos_token_id:
                logging.warning('encountered eos_token_id, stopping after encoding %d bits' % i)
                break

            # For text->bits->text
            partial = enc.decode(output[len(context):].tolist(), skip_special_tokens=True)
            if '<eos>' in partial:
                break

    avg_NLL = -total_log_probs / total_num_for_stats
    avg_KL = total_kl / total_num_for_stats
    avg_Hq = total_entropy_ptau / total_num_for_stats
    words_per_bit = total_num_for_stats / i
    stats: Dict[str, object] = {"encoded_bits_in_output": encoded_bits_in_output}

    return output[len(context):].tolist(), avg_NLL, avg_KL, words_per_bit, avg_Hq, stats, message[i:]


def decode_conversation_meteor(model, enc, text, context, key, nonce, device='cuda', temp=1.0, precision=16, topk=50000,
                               is_sort=False, randomized=False) -> Tuple[List[bool], List[str]]:
    import torch
    # inp is a list of token indices
    # context is a list of token indices
    inp = enc.encode(text)

    mask_generator = DRBG(key, sample_seed_prefix + nonce)
    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)

    max_val = 2 ** precision
    threshold = 2 ** (-precision)
    cur_interval = [0, max_val]  # bottom inclusive, top exclusive

    prev = context
    past = None
    message = []
    debug_entropies = []
    debug_encoded_num = []
    with torch.no_grad():
        i = 0
        while i < len(inp):
            result = model(prev.unsqueeze(0), past_key_values=past)
            logits = result.logits
            past = result.past_key_values
            past = limit_past(past)
            logits[0, -1, -1] = -1e20  # endoftext can't happen
            logits[0, -1, 628] = -1e20  # 2 newlines can't happen
            logits, indices = logits[0, -1, :].sort(descending=True)
            logits = logits.double()
            logits_temp = logits / temp
            log_probs_temp = F.log_softmax(logits_temp, dim=0)
            probs_temp = F.softmax(logits_temp, dim=0)

            # Cutoff low probabilities that would be rounded to 0
            cur_int_range = cur_interval[1] - cur_interval[0]
            cur_threshold = 1 / cur_int_range
            k = min(max(2, (probs_temp < cur_threshold).nonzero()[0].item()), topk)
            probs_temp_int = probs_temp[:k]  # Cutoff all but top k

            # Rescale to correct range
            probs_temp_int = probs_temp_int / probs_temp_int.sum() * cur_int_range
            entropy_in_this_distribution = entropy(probs_temp, log_probs_temp)

            # Round probabilities to integers given precision
            probs_temp_int = probs_temp_int.round().long()
            if is_sort:
                probs_temp_int, indices = bin_sort(probs_temp_int, indices, cur_int_range, entropy_in_this_distribution,
                                                   device)
            cum_probs = probs_temp_int.cumsum(0)

            # Remove any elements from the bottom if rounding caused the total prob to be too large
            overfill_index = (cum_probs > cur_int_range).nonzero()
            if len(overfill_index) > 0:
                cum_probs = cum_probs[:overfill_index[0]]
                k = overfill_index[0].item()

            # Add any mass to the top if removing/rounding causes the total prob to be too small
            cum_probs += cur_int_range - cum_probs[-1]  # add

            # Covnert to position in range
            cum_probs += cur_interval[0]

            rank = (indices == inp[i]).nonzero().item()

            # Handle most errors that could happen because of BPE with heuristic
            if rank >= k:
                true_token_text = enc.decoder[inp[i]]
                for rank_idx in range(k):
                    prop_token_text = enc.decoder[indices[rank_idx].item()]
                    # common case that is not caught
                    if inp[i] == 128 and indices[rank_idx] == 198:
                        rank = rank_idx
                        inp[i] = indices[rank_idx].item()
                        break

                    # Is there a more likely prefix token that could be the actual token generated?
                    if len(prop_token_text) <= len(true_token_text) and \
                            prop_token_text == true_token_text[:len(prop_token_text)]:
                        rank = rank_idx
                        suffix = true_token_text[len(prop_token_text):]
                        suffix_tokens = enc.encode(suffix)  # a list
                        inp[i] = indices[rank_idx].item()
                        inp[i + 1:i + 1] = suffix_tokens  # insert suffix tokens into list
                        break

                    # Is there a more likely longer token that could be the actual token generated?
                    elif len(prop_token_text) > len(true_token_text) and \
                            true_token_text == prop_token_text[:len(true_token_text)]:
                        whole_text = true_token_text
                        num_extra = 1
                        while len(whole_text) < len(prop_token_text):
                            whole_text += enc.decoder[inp[i + num_extra]]
                            num_extra += 1
                        if prop_token_text == whole_text[:len(prop_token_text)]:
                            rank = rank_idx
                            inp[i] = indices[rank_idx].item()
                            for j in range(1, num_extra):
                                del inp[i + j]

                            if len(whole_text) > len(prop_token_text):
                                suffix = whole_text[len(prop_token_text):]
                                suffix_tokens = enc.encode(suffix)  # a list
                                inp[i + 1:i + 1] = suffix_tokens  # insert suffix tokens into list
                            break
                else:
                    logging.warning('Unable to fix BPE error: token received: %s=%d, text: %s' % (true_token_text, inp[i], text))
                    rank = 0

            selection = rank

            # Calculate new range as ints
            new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]
            new_int_top = cum_probs[selection]

            # Convert range to bits
            new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
            new_int_top_bits_inc = list(
                reversed(int2bits(new_int_top - 1, precision)))  # -1 here because upper bound is exclusive

            # Emit most significant bits which are now fixed and update interval
            num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
            if i == len(inp) - 1:
                new_bits = new_int_bottom_bits_inc
            else:
                new_bits = new_int_top_bits_inc[:num_bits_encoded]

            debug_entropies.append(entropy_in_this_distribution)
            debug_encoded_num.append(num_bits_encoded)

            # Get the mask and apply it to the recovered bits
            mask_bits = mask_generator.generate_bits(precision)
            for b in range(0, len(new_bits)):
                new_bits[b] = new_bits[b] ^ mask_bits[b]
            message += new_bits

            # Update history with new token
            prev = torch.tensor([inp[i]], device=device, dtype=torch.long)

            i += 1

    return message, enc.decode(inp)[1]


def cumsum_adjust(probs_temp_int, precision):
    max_val = 2**precision
    cum_probs = probs_temp_int.cumsum(0)

    # Remove any elements from the bottom if rounding caused the total prob to be too large
    overfill_index = (cum_probs > max_val).nonzero()
    if len(overfill_index) > 0:
        cum_probs = cum_probs[:overfill_index[0]]

    # Add any mass to the top if removing/rounding causes the total prob to be too small
    cum_probs += max_val - cum_probs[-1]  # add

    # Get out resulting probabilities
    probs_final = cum_probs.clone()
    probs_final[1:] = cum_probs[1:] - cum_probs[:-1]

    return cum_probs


@dataclass
class TokenProbabilities:
    indices: torch.Tensor
    past: List[Tuple[torch.FloatTensor]]
    probs_temp_int: torch.Tensor
    log_probs: torch.Tensor

    def __iter__(self):
        return iter(astuple(self))


def get_token_probabilities(model: GPT2LMHeadModel, context: Optional[torch.LongTensor], past_key_values: Optional[Tuple[Tuple[torch.Tensor]]], temp: float, topk: int, precision: int, sort: bool, device: str) -> TokenProbabilities:
    max_val = 2 ** precision
    cur_interval = [0, max_val]  # bottom inclusive, top exclusive

    result = model(context.unsqueeze(0), past_key_values=past_key_values)
    logits = result.logits
    past = result.past_key_values
    past = limit_past(past)
    # logits[0, -1, -1] = -1e20  # endoftext token can't happen
    logits[0, -1, 628] = -1e20  # 2 newlines token can't happen
    logits, indices = logits[0, -1, :].sort(descending=True)
    logits = logits.double()
    logits_temp = logits / temp
    probs_temp = F.softmax(logits_temp, dim=0)
    log_probs_temp = F.log_softmax(logits_temp, dim=0)
    log_probs = F.log_softmax(logits, dim=0)
    # Cutoff low probabilities that would be rounded to 0
    cur_int_range = cur_interval[1] - cur_interval[0]
    cur_threshold = 1 / cur_int_range
    k = min(max(2, (probs_temp < cur_threshold).nonzero()[0].item()), topk)
    probs_temp_int = probs_temp[:k]  # Cutoff all but top k
    indices = indices[:k]

    # Rescale to correct range
    probs_temp_int = probs_temp_int / probs_temp_int.sum() * cur_int_range

    # Round probabilities to integers given precision
    probs_temp_int = probs_temp_int.round().long()

    if sort:
        entropy_in_this_distribution = entropy(probs_temp, log_probs_temp)
        probs_temp_int, indices = bin_sort(probs_temp_int, indices, cur_int_range,
                                           entropy_in_this_distribution, device)

    return TokenProbabilities(indices, past, probs_temp_int, log_probs)


def sort_tokens(enc, indices, probs_int):
    s = sorted(zip(indices, probs_int), key=lambda x: enc.decode(x[0].view(1)))
    indices, probs = torch.tensor([a for a, _ in s]), torch.tensor([b for _, b in s])
    return indices, probs


def encrypt(message_bits, mask_generator, precision):
    # add padding
    message_bits = message_bits + [0] * (precision - len(message_bits))
    mask_bits = mask_generator.generate_bits(precision)
    for b in range(0, precision):
        message_bits[b] = message_bits[b] ^ mask_bits[b]
    assert len(message_bits) == precision
    return message_bits


def encode_meteor_binned_resample(model: GPT2LMHeadModel,
                                  enc: GPT2Tokenizer,
                                  message: bytes,
                                  context: List[int],
                                  key: bytes,
                                  nonce: bytes,
                                  finish_sent: bool,
                                  device: Literal['cuda', 'cpu'],
                                  temp: float = 1.0,
                                  precision: int = 16,
                                  topk: int = 50000,
                                  is_sort: bool = False):
    logging.debug(f'will embed message {message}, {len(message)} bytes, precision {precision}')
    x = message
    message = bitarray.bitarray()
    message.frombytes(x)
    mask_generator = DRBG(key, sample_seed_prefix + nonce)
    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)
    context_len = len(context)

    prev = context
    output = context
    encoded_bits_in_output = []
    past = None

    total_num = 0
    total_num_for_stats = 0
    total_log_probs = 0
    total_kl = 0  # in bits
    total_entropy_ptau = 0

    trie = TokenTrie.from_tokenizer(enc)
    entropies_for_stats = []

    with torch.no_grad():
        i = 0
        sent_finish = False
        while i < len(message) or (finish_sent and not sent_finish):
            logging.debug(f'{total_num}: prev = {prev}')
            indices, past, probs_int, log_probs = get_token_probabilities(model=model, context=prev,
                                                                          past_key_values=past, temp=temp,
                                                                          topk=topk, precision=precision,
                                                                          sort=is_sort, device=device)
            trie.update(zip(indices, probs_int))
            reprs, tokens, probs = zip(*trie.distribution())
            probs = torch.tensor(probs, device=device)
            cum_probs = cumsum_adjust(probs, precision=precision)
            # conditions for having reached the end of the message
            if i >= len(message):
                # select first message in distribution until finish
                selection = 0
                sent_finish = is_sent_finish(tokens[selection].item(), enc)
            else:
                # Apply the mask to the message
                message_bits = message[i:i + precision]
                if i + precision > len(message):
                    message_bits = message_bits + [0] * (i + precision - len(message))

                mask_bits = mask_generator.generate_bits(precision)

                for b in range(0, len(message_bits)):
                    message_bits[b] = message_bits[b] ^ mask_bits[b]

                # Get selected index based on binary fraction from message bits
                message_idx = bits2int(reversed(message_bits))
                selection = (cum_probs > message_idx).nonzero()[0].item()

                # Calculate new range as ints
                new_int_bottom = cum_probs[selection - 1] if selection > 0 else 0
                new_int_top = cum_probs[selection]

                # Convert range to bits
                new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
                new_int_top_bits_inc = list(
                    reversed(int2bits(new_int_top - 1, precision)))  # -1 here because upper bound is exclusive

                # Consume most significant bits which are now fixed and update interval
                num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
                encoded_bits_in_output += [num_bits_encoded]  # for statistics
                i += num_bits_encoded
                logging.debug(
                    f'{total_num}:{num_bits_encoded} bits embedded: {message_bits[:num_bits_encoded]} in [{new_int_bottom_bits_inc},{new_int_top_bits_inc}]')

                # Gather statistics
                total_log_probs += log_probs[selection].item()

                q = cum_probs.double() / cum_probs.sum()
                logq = q.log()
                total_kl += kl(q, logq, log_probs[:len(q)])
                entr = entropy(probs / probs.sum(), torch.log(probs / probs.sum()))
                total_entropy_ptau += entr
                entropies_for_stats.append(entr)
                total_num_for_stats += 1
                logging.debug(f'average entropy: {total_entropy_ptau / total_num_for_stats}')

            # Update history with new token
            if len(tokens[selection]) > 1:
                # we chose an undecodable path. Resample from subtree
                repr = reprs[selection]
                st = trie.subtree(repr)
                if st is None:
                    logging.debug(trie.visualize(max_depth=2))
                    raise Exception(f'no subtrie for repr {repr} found')
                tokens = st.tokens()
                probabilities = torch.tensor(st.probabilities())
                cum_probs = cumsum_adjust(probabilities, precision)
                mask = mask_generator.generate_bits(precision)
                message_idx = bits2int(reversed(mask))
                selection = (cum_probs > message_idx).nonzero()[0].item()
                prev = torch.tensor(tokens[selection], device=device).view(1)
                logging.debug(
                    f'{total_num}: resampled {prev} = {enc.decode(prev)[0].encode("utf-8", errors=enc.errors)} from subtrie {repr}')
            else:
                prev = torch.tensor(tokens[selection], device=device).view(1)
                logging.debug(
                    f'{total_num}: resampled from singleton {prev} = {enc.decode(prev)[0].encode("utf-8", errors=enc.errors)}')
            output = torch.cat((output, prev))
            total_num += 1

            # For text->bits->text
            partial = enc.decode(output[context_len:].tolist())[0]
            if prev == enc.eos_token_id or '<eos>' in partial:
                break
    logging.debug(f'average entropy: {total_entropy_ptau/total_num_for_stats}')

    avg_NLL = -total_log_probs / total_num_for_stats
    avg_KL = total_kl / total_num_for_stats
    avg_Hq = total_entropy_ptau / total_num_for_stats
    words_per_bit = total_num_for_stats / i
    stats: Dict[str, object] = {"encoded_bits_in_output": encoded_bits_in_output, "entropies": entropies_for_stats}

    return output[context_len:].tolist(), avg_NLL, avg_KL, words_per_bit, avg_Hq, stats


def decode_meteor_binned_resample(model: GPT2LMHeadModel,
                                  enc: GPT2Tokenizer,
                                  text: str,
                                  context,
                                  key: bytes,
                                  nonce: bytes,
                                  device: Literal['cuda', 'cpu'] = 'cuda',
                                  temp: float = 1.0,
                                  precision: int = 16,
                                  topk: int = 50000,
                                  is_sort: bool = False,
                                  enc_tokens: List[int] = None) -> List[bool]:
    # TODO enc_tokens is for debug only, remove!
    import torch
    # inp is a list of token indices
    # context is a list of token indices
    inp: bytes = text.encode('utf-8', errors=enc.errors)

    trie = TokenTrie.from_tokenizer(enc)

    mask_generator = DRBG(key, sample_seed_prefix + nonce)
    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)

    max_val = 2 ** precision
    threshold = 2 ** (-precision)
    cur_interval = [0, max_val]  # bottom inclusive, top exclusive

    prev = context
    past = None
    message = []
    debug_entropies = []
    debug_encoded_num = []
    with torch.no_grad():
        i = 0
        while inp:
            logging.debug(f'{i}: prev = {prev}')
            indices, past, probs_int, _ = get_token_probabilities(model=model, context=prev, past_key_values=past,
                                                                  temp=temp, topk=topk, precision=precision,
                                                                  sort=bin_sort if is_sort else None, device=device)
            # indices, probs_int = sort_tokens(enc, indices, probs_int)
            # cum_probs = cumsum_adjust(probs_int, precision=precision)
            trie.update(zip(indices, probs_int))
            reprs, tokens, probs = zip(*trie.distribution())
            probs = torch.tensor(probs, device=device)
            #if is_sort:
            #    probs_temp_int, indices = bin_sort(probs_temp_int, indices, cur_int_range, entropy_in_this_distribution,
            #                                       device)

            cum_probs = cumsum_adjust(probs, precision)

            selected = enc.encode(inp.decode('utf-8', errors=enc.errors))[0]
            rank = None
            for rank, toks in enumerate(tokens):
                if selected in toks:
                    break
            assert selected in tokens[rank], f'{selected} not in {tokens[rank]}: {tokens}'

            new_int_bottom = cum_probs[rank - 1] if rank > 0 else cur_interval[0]
            new_int_top = cum_probs[rank]

            # Convert range to bits
            new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
            new_int_top_bits_inc = list(
                reversed(int2bits(new_int_top - 1, precision)))  # -1 here because upper bound is exclusive

            # Emit most significant bits which are now fixed and update interval
            num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
            new_bits = new_int_top_bits_inc[:num_bits_encoded]

            logging.debug(f'{i}:{num_bits_encoded} bits recovered: {new_bits[:num_bits_encoded]} in [{new_int_bottom_bits_inc},{new_int_top_bits_inc}]')

            # Get the mask and apply it to the recovered bits
            mask_bits = mask_generator.generate_bits(precision)
            for b in range(0, len(new_bits)):
                new_bits[b] = new_bits[b] ^ mask_bits[b]
            message += new_bits

            if len(tokens[rank]) > 1:
                # during encoding, resampling was done. Generate bits and update tokenization
                repr = reprs[rank]
                st = trie.subtree(repr)
                tokens = st.tokens()
                probabilities = torch.tensor(st.probabilities(), device=device)
                cum_probs = cumsum_adjust(probabilities, precision)
                mask = mask_generator.generate_bits(precision)
                message_idx = bits2int(reversed(mask))
                selection = (cum_probs > message_idx).nonzero()[0].item()
                resampled_token = tokens[selection]
                resampled_token_str: bytes = enc.decoder[resampled_token].encode('utf-8', errors=enc.errors)
                logging.debug(f'resampled {resampled_token} = "{resampled_token_str}" from subtrie {repr}')
                selected = resampled_token
            else:
                logging.debug(f'resampled from singleton {tokens[rank][0]}')
                selected = tokens[rank][0]

            # Update history with new token
            inp = inp[len(enc.decode(selected)[0].encode('utf-8', errors=enc.errors)):]
            prev = torch.tensor([selected], device=device, dtype=torch.long)
            if enc_tokens is not None:
                assert prev == enc_tokens[i], f'{prev} != {enc_tokens[i]}'
            i += 1

    return message


# <editor-fold desc="CTR randomized">
def encrypt_ctr_aes(key: bytes, message: bytes) -> bytes:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CTR(iv))
    encryptor = cipher.encryptor()
    ct = encryptor.update(message) + encryptor.finalize()
    return iv.zfill(16) + ct


def decrypt_ctr_aes(key: bytes, ciphertext: bytes) -> bytes:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    iv = ciphertext[0:16]
    cipher = Cipher(algorithms.AES(key), modes.CTR(iv))
    decryptor = cipher.decryptor()
    message = decryptor.update(ciphertext[16:]) + decryptor.finalize()
    return message


def encode_meteor_randomized(model, enc, message, context: List[int], key, nonce, finish_sent=False, device='cuda',
                             temp=1.0,
                             precision=16, topk=50000, is_sort=False):
    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)

    max_val = 2 ** precision
    threshold = 2 ** (-precision)
    cur_interval = [0, max_val]  # bottom inclusive, top exclusive

    prev = context
    output = context
    encoded_bits_in_output = []
    past = None

    total_num = 0
    total_num_for_stats = 0
    total_log_probs = 0
    total_kl = 0  # in bits
    total_entropy_ptau = 0

    ciphertext = encrypt_ctr_aes(key, message)  # ciphertext is indistinguishable from random
    ciphertext_bits = bitarray.bitarray()
    ciphertext_bits.frombytes(ciphertext)
    ciphertext_bits = ciphertext_bits.tolist()

    with torch.no_grad():
        i = 0
        sent_finish = False
        while (i < len(ciphertext_bits) or (finish_sent and not sent_finish)):
            result = model(prev.unsqueeze(0), past_key_values=past)
            logits = result.logits
            past = result.past_key_values
            past = limit_past(past)
            logits[0, -1, -1] = -1e20  # endoftext token can't happen
            logits[0, -1, 628] = -1e20  # 2 newlines token can't happen
            logits, indices = logits[0, -1, :].sort(descending=True)
            logits = logits.double()
            logits_temp = logits / temp
            probs_temp = F.softmax(logits_temp, dim=0)
            log_probs_temp = F.log_softmax(logits_temp, dim=0)
            log_probs = F.log_softmax(logits, dim=0)

            # conditions for having reached the end of the message
            if i >= len(ciphertext_bits):
                selection = 0
                sent_finish = is_sent_finish(indices[selection].item(), enc)
            else:
                # Cutoff low probabilities that would be rounded to 0
                cur_int_range = cur_interval[1] - cur_interval[0]
                cur_threshold = 1 / cur_int_range
                k = min(max(2, (probs_temp < cur_threshold).nonzero()[0].item()), topk)
                probs_temp_int = probs_temp[:k]  # Cutoff all but top k
                old_indices = indices
                indices = indices[:k]

                # Rescale to correct range
                probs_temp_int = probs_temp_int / probs_temp_int.sum() * cur_int_range

                entropy_in_this_distribution = entropy(probs_temp, log_probs_temp)
                #logging.debug('%d: %.02f, H(D)=%.02f' % (i, i / float(len(ciphertext_bits)), entropy_in_this_distribution))

                # Round probabilities to integers given precision
                probs_temp_int = probs_temp_int.round().long()

                if is_sort:
                    probs_temp_int, indices = bin_sort(probs_temp_int, indices, cur_int_range,
                                                       entropy_in_this_distribution, device)
                cum_probs = probs_temp_int.cumsum(0)

                # Remove any elements from the bottom if rounding caused the total prob to be too large
                overfill_index = (cum_probs > cur_int_range).nonzero()
                if len(overfill_index) > 0:
                    cum_probs = cum_probs[:overfill_index[0]]

                # Add any mass to the top if removing/rounding causes the total prob to be too small
                cum_probs += cur_int_range - cum_probs[-1]  # add

                # Get out resulting probabilities
                probs_final = cum_probs.clone()
                probs_final[1:] = cum_probs[1:] - cum_probs[:-1]

                # Convert to position in range
                cum_probs += cur_interval[0]

                # Apply the mask to the message
                # message_bits = message[i:i + precision]
                # if i + precision > len(message):
                #    message_bits = message_bits + [0] * (i + precision - len(message))

                # mask_bits = mask_generator.generate_bits(precision)

                # for b in range(0, len(message_bits)):
                #    message_bits[b] = message_bits[b] ^ mask_bits[b]

                # Get selected index based on binary fraction from message bits
                message_idx = bits2int(reversed(ciphertext_bits[i:i + precision]))
                selection = (cum_probs > message_idx).nonzero()[0].item()

                # Calculate new range as ints
                new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]
                new_int_top = cum_probs[selection]

                # Convert range to bits
                new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
                new_int_top_bits_inc = list(
                    reversed(int2bits(new_int_top - 1, precision)))  # -1 here because upper bound is exclusive

                # Consume most significant bits which are now fixed and update interval
                num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
                encoded_bits_in_output += [num_bits_encoded]  # for statistics
                i += num_bits_encoded

                # Gather statistics
                total_log_probs += log_probs[selection].item()

                q = probs_final.double() / probs_final.sum()
                logq = q.log()
                total_kl += kl(q, logq, log_probs[:len(q)])
                total_entropy_ptau += entropy_in_this_distribution
                total_num_for_stats += 1

            # Update history with new token
            prev = indices[selection].view(1)
            output = torch.cat((output, prev))
            total_num += 1

            # For text->bits->text
            partial = enc.decode(output[len(context):].tolist())
            if '<eos>' in partial:
                break

    avg_NLL = -total_log_probs / total_num_for_stats
    avg_KL = total_kl / total_num_for_stats
    avg_Hq = total_entropy_ptau / total_num_for_stats
    words_per_bit = total_num_for_stats / i
    stats: Dict[str, object] = {"encoded_bits_in_output": encoded_bits_in_output}

    return output[len(context):].tolist(), avg_NLL, avg_KL, words_per_bit, avg_Hq, stats


def decode_meteor_randomized(model, enc, text, context, key, nonce, device='cuda', temp=1.0, precision=16, topk=50000,
                             is_sort=False) -> Tuple[List[bool], List[str]]:
    import torch
    # inp is a list of token indices
    # context is a list of token indices
    inp = enc.encode(text)

    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)

    max_val = 2 ** precision
    threshold = 2 ** (-precision)
    cur_interval = [0, max_val]  # bottom inclusive, top exclusive

    prev = context
    past = None
    ciphertext_bits = []
    debug_entropies = []
    debug_encoded_num = []
    with torch.no_grad():
        i = 0
        while i < len(inp):
            result = model(prev.unsqueeze(0), past_key_values=past)
            logits = result.logits
            past = result.past_key_values
            past = limit_past(past)
            logits[0, -1, -1] = -1e20  # endoftext can't happen
            logits[0, -1, 628] = -1e20  # 2 newlines can't happen
            logits, indices = logits[0, -1, :].sort(descending=True)
            logits = logits.double()
            logits_temp = logits / temp
            log_probs_temp = F.log_softmax(logits_temp, dim=0)
            probs_temp = F.softmax(logits_temp, dim=0)

            # Cutoff low probabilities that would be rounded to 0
            cur_int_range = cur_interval[1] - cur_interval[0]
            cur_threshold = 1 / cur_int_range
            k = min(max(2, (probs_temp < cur_threshold).nonzero()[0].item()), topk)
            probs_temp_int = probs_temp[:k]  # Cutoff all but top k

            # Rescale to correct range
            probs_temp_int = probs_temp_int / probs_temp_int.sum() * cur_int_range
            entropy_in_this_distribution = entropy(probs_temp, log_probs_temp)

            # Round probabilities to integers given precision
            probs_temp_int = probs_temp_int.round().long()
            if is_sort:
                probs_temp_int, indices = bin_sort(probs_temp_int, indices, cur_int_range, entropy_in_this_distribution,
                                                   device)
            cum_probs = probs_temp_int.cumsum(0)

            # Remove any elements from the bottom if rounding caused the total prob to be too large
            overfill_index = (cum_probs > cur_int_range).nonzero()
            if len(overfill_index) > 0:
                cum_probs = cum_probs[:overfill_index[0]]
                k = overfill_index[0].item()

            # Add any mass to the top if removing/rounding causes the total prob to be too small
            cum_probs += cur_int_range - cum_probs[-1]  # add

            # Covnert to position in range
            cum_probs += cur_interval[0]

            rank = (indices == inp[i]).nonzero().item()

            # Handle most errors that could happen because of BPE with heuristic
            if rank >= k:
                true_token_text = enc.decoder[inp[i]]
                for rank_idx in range(k):
                    prop_token_text = enc.decoder[indices[rank_idx].item()]
                    # common case that is not caught
                    if inp[i] == 128 and indices[rank_idx] == 198:
                        rank = rank_idx
                        inp[i] = indices[rank_idx].item()
                        break

                    # Is there a more likely prefix token that could be the actual token generated?
                    if len(prop_token_text) <= len(true_token_text) and \
                            prop_token_text == true_token_text[:len(prop_token_text)]:
                        rank = rank_idx
                        suffix = true_token_text[len(prop_token_text):]
                        suffix_tokens = enc.encode(suffix)  # a list
                        inp[i] = indices[rank_idx].item()
                        inp[i + 1:i + 1] = suffix_tokens  # insert suffix tokens into list
                        break

                    # Is there a more likely longer token that could be the actual token generated?
                    elif len(prop_token_text) > len(true_token_text) and \
                            true_token_text == prop_token_text[:len(true_token_text)]:
                        whole_text = true_token_text
                        num_extra = 1
                        while len(whole_text) < len(prop_token_text):
                            whole_text += enc.decoder[inp[i + num_extra]]
                            num_extra += 1
                        if prop_token_text == whole_text[:len(prop_token_text)]:
                            rank = rank_idx
                            inp[i] = indices[rank_idx].item()
                            for j in range(1, num_extra):
                                del inp[i + j]

                            if len(whole_text) > len(prop_token_text):
                                suffix = whole_text[len(prop_token_text):]
                                suffix_tokens = enc.encode(suffix)  # a list
                                inp[i + 1:i + 1] = suffix_tokens  # insert suffix tokens into list
                            break
                else:
                    logging.warning('Unable to fix BPE error: token received: %s=%d, text: %s' % (true_token_text, inp[i], text))
                    rank = 0

            selection = rank

            # Calculate new range as ints
            new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]
            new_int_top = cum_probs[selection]

            # Convert range to bits
            new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
            new_int_top_bits_inc = list(
                reversed(int2bits(new_int_top - 1, precision)))  # -1 here because upper bound is exclusive

            # Emit most significant bits which are now fixed and update interval
            num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
            if i == len(inp) - 1:
                new_bits = new_int_bottom_bits_inc
            else:
                new_bits = new_int_top_bits_inc[:num_bits_encoded]

            debug_entropies.append(entropy_in_this_distribution)
            debug_encoded_num.append(num_bits_encoded)

            # Get the mask and apply it to the recovered bits
            # mask_bits = mask_generator.generate_bits(precision)
            # for b in range(0, len(new_bits)):
            #    new_bits[b] = new_bits[b] ^ mask_bits[b]
            # message += new_bits
            ciphertext_bits += new_bits

            # Update history with new token
            prev = torch.tensor([inp[i]], device=device, dtype=torch.long)

            i += 1

    ciphertext = bitarray.bitarray(ciphertext_bits).tobytes()
    message = bitarray.bitarray()
    message.frombytes(decrypt_ctr_aes(key, ciphertext))

    return message.tolist(), enc.decode(inp)[1]
# </editor-fold>


def encode_meteor(model, enc, message: bytes, context: List[int], key, nonce, finish_sent=False, device='cuda',
                  temp=1.0,
                  precision=16, topk=50000, is_sort=False):
    x = message
    message = bitarray.bitarray()
    message.frombytes(x)
    mask_generator = DRBG(key, sample_seed_prefix + nonce)
    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)

    max_val = 2 ** precision
    threshold = 2 ** (-precision)
    cur_interval = [0, max_val]  # bottom inclusive, top exclusive

    prev = context
    output = context
    encoded_bits_in_output = []
    past = None

    total_num = 0
    total_num_for_stats = 0
    total_log_probs = 0
    total_kl = 0  # in bits
    total_entropy_ptau = 0

    with torch.no_grad():
        i = 0
        sent_finish = False
        while (i < len(message) or (finish_sent and not sent_finish)):
            # logging.debug(f'{i}: {i / float(len(message))}')
            result = model(prev.unsqueeze(0), past_key_values=past)
            logits = result.logits
            past = result.past_key_values
            past = limit_past(past)
            logits[0, -1, -1] = -1e20  # endoftext token can't happen
            logits[0, -1, 628] = -1e20  # 2 newlines token can't happen
            logits, indices = logits[0, -1, :].sort(descending=True)
            logits = logits.double()
            logits_temp = logits / temp
            probs_temp = F.softmax(logits_temp, dim=0)
            log_probs_temp = F.log_softmax(logits_temp, dim=0)
            log_probs = F.log_softmax(logits, dim=0)

            # conditions for having reached the end of the message
            if i >= len(message):
                selection = 0
                sent_finish = is_sent_finish(indices[selection].item(), enc)
            else:
                # Cutoff low probabilities that would be rounded to 0
                cur_int_range = cur_interval[1] - cur_interval[0]
                cur_threshold = 1 / cur_int_range
                k = min(max(2, (probs_temp < cur_threshold).nonzero()[0].item()), topk)
                probs_temp_int = probs_temp[:k]  # Cutoff all but top k
                old_indices = indices
                indices = indices[:k]

                # Rescale to correct range
                probs_temp_int = probs_temp_int / probs_temp_int.sum() * cur_int_range

                entropy_in_this_distribution = entropy(probs_temp, log_probs_temp)

                # Round probabilities to integers given precision
                probs_temp_int = probs_temp_int.round().long()

                if is_sort:
                    probs_temp_int, indices = bin_sort(probs_temp_int, indices, cur_int_range,
                                                       entropy_in_this_distribution, device)
                cum_probs = probs_temp_int.cumsum(0)

                # Remove any elements from the bottom if rounding caused the total prob to be too large
                overfill_index = (cum_probs > cur_int_range).nonzero()
                if len(overfill_index) > 0:
                    cum_probs = cum_probs[:overfill_index[0]]

                # Add any mass to the top if removing/rounding causes the total prob to be too small
                cum_probs += cur_int_range - cum_probs[-1]  # add

                # Get out resulting probabilities
                probs_final = cum_probs.clone()
                probs_final[1:] = cum_probs[1:] - cum_probs[:-1]

                # Convert to position in range
                cum_probs += cur_interval[0]

                # Apply the mask to the message
                message_bits = message[i:i + precision]
                if i + precision > len(message):
                    message_bits = message_bits + [0] * (i + precision - len(message))

                mask_bits = mask_generator.generate_bits(precision)

                for b in range(0, len(message_bits)):
                    message_bits[b] = message_bits[b] ^ mask_bits[b]

                # Get selected index based on binary fraction from message bits
                message_idx = bits2int(reversed(message_bits))
                selection = (cum_probs > message_idx).nonzero()[0].item()

                # Calculate new range as ints
                new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]
                new_int_top = cum_probs[selection]

                # Convert range to bits
                new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
                new_int_top_bits_inc = list(
                    reversed(int2bits(new_int_top - 1, precision)))  # -1 here because upper bound is exclusive

                # Consume most significant bits which are now fixed and update interval
                num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
                encoded_bits_in_output += [num_bits_encoded]  # for statistics
                i += num_bits_encoded

                # Gather statistics
                total_log_probs += log_probs[selection].item()

                q = probs_final.double() / probs_final.sum()
                logq = q.log()
                total_kl += kl(q, logq, log_probs[:len(q)])
                total_entropy_ptau += entropy_in_this_distribution
                total_num_for_stats += 1

            # Update history with new token
            prev = indices[selection].view(1)
            output = torch.cat((output, prev))
            total_num += 1

            # For text->bits->text
            partial = enc.decode(output[len(context):].tolist())
            if '<eos>' in partial:
                break

    avg_NLL = -total_log_probs / total_num_for_stats
    avg_KL = total_kl / total_num_for_stats
    avg_Hq = total_entropy_ptau / total_num_for_stats
    words_per_bit = total_num_for_stats / i
    stats: Dict[str, object] = {"encoded_bits_in_output": encoded_bits_in_output}

    return output[len(context):].tolist(), avg_NLL, avg_KL, words_per_bit, avg_Hq, stats


def decode_meteor(model, enc, text, context, key, nonce, device='cuda', temp=1.0, precision=16, topk=50000,
                  is_sort=False) -> Tuple[List[bool], List[str]]:
    import torch
    # inp is a list of token indices
    # context is a list of token indices
    inp = enc.encode(text)

    mask_generator = DRBG(key, sample_seed_prefix + nonce)
    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)

    max_val = 2 ** precision
    threshold = 2 ** (-precision)
    cur_interval = [0, max_val]  # bottom inclusive, top exclusive

    prev = context
    past = None
    message = []
    debug_entropies = []
    debug_encoded_num = []
    with torch.no_grad():
        i = 0
        while i < len(inp):
            result = model(prev.unsqueeze(0), past_key_values=past)
            logits = result.logits
            past = result.past_key_values
            past = limit_past(past)
            logits[0, -1, -1] = -1e20  # endoftext can't happen
            logits[0, -1, 628] = -1e20  # 2 newlines can't happen
            logits, indices = logits[0, -1, :].sort(descending=True)
            logits = logits.double()
            logits_temp = logits / temp
            log_probs_temp = F.log_softmax(logits_temp, dim=0)
            probs_temp = F.softmax(logits_temp, dim=0)

            # Cutoff low probabilities that would be rounded to 0
            cur_int_range = cur_interval[1] - cur_interval[0]
            cur_threshold = 1 / cur_int_range
            k = min(max(2, (probs_temp < cur_threshold).nonzero()[0].item()), topk)
            probs_temp_int = probs_temp[:k]  # Cutoff all but top k

            # Rescale to correct range
            probs_temp_int = probs_temp_int / probs_temp_int.sum() * cur_int_range
            entropy_in_this_distribution = entropy(probs_temp, log_probs_temp)

            # Round probabilities to integers given precision
            probs_temp_int = probs_temp_int.round().long()
            if is_sort:
                probs_temp_int, indices = bin_sort(probs_temp_int, indices, cur_int_range, entropy_in_this_distribution,
                                                   device)
            cum_probs = probs_temp_int.cumsum(0)

            # Remove any elements from the bottom if rounding caused the total prob to be too large
            overfill_index = (cum_probs > cur_int_range).nonzero()
            if len(overfill_index) > 0:
                cum_probs = cum_probs[:overfill_index[0]]
                k = overfill_index[0].item()

            # Add any mass to the top if removing/rounding causes the total prob to be too small
            cum_probs += cur_int_range - cum_probs[-1]  # add

            # Covnert to position in range
            cum_probs += cur_interval[0]

            rank = (indices == inp[i]).nonzero().item()

            # Handle most errors that could happen because of BPE with heuristic
            if rank >= k:
                true_token_text = enc.decoder[inp[i]]
                for rank_idx in range(k):
                    prop_token_text = enc.decoder[indices[rank_idx].item()]
                    # common case that is not caught
                    if inp[i] == 128 and indices[rank_idx] == 198:
                        rank = rank_idx
                        inp[i] = indices[rank_idx].item()
                        break

                    # Is there a more likely prefix token that could be the actual token generated?
                    if len(prop_token_text) <= len(true_token_text) and \
                            prop_token_text == true_token_text[:len(prop_token_text)]:
                        rank = rank_idx
                        suffix = true_token_text[len(prop_token_text):]
                        suffix_tokens = enc.encode(suffix)  # a list
                        inp[i] = indices[rank_idx].item()
                        inp[i + 1:i + 1] = suffix_tokens  # insert suffix tokens into list
                        break

                    # Is there a more likely longer token that could be the actual token generated?
                    elif len(prop_token_text) > len(true_token_text) and \
                            true_token_text == prop_token_text[:len(true_token_text)]:
                        whole_text = true_token_text
                        num_extra = 1
                        while len(whole_text) < len(prop_token_text):
                            whole_text += enc.decoder[inp[i + num_extra]]
                            num_extra += 1
                        if prop_token_text == whole_text[:len(prop_token_text)]:
                            rank = rank_idx
                            inp[i] = indices[rank_idx].item()
                            for j in range(1, num_extra):
                                del inp[i + j]

                            if len(whole_text) > len(prop_token_text):
                                suffix = whole_text[len(prop_token_text):]
                                suffix_tokens = enc.encode(suffix)  # a list
                                inp[i + 1:i + 1] = suffix_tokens  # insert suffix tokens into list
                            break
                else:
                    logging.warning('Unable to fix BPE error: token received: %s=%d, text: %s' % (true_token_text, inp[i], text))
                    rank = 0

            selection = rank

            # Calculate new range as ints
            new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]
            new_int_top = cum_probs[selection]

            # Convert range to bits
            new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
            new_int_top_bits_inc = list(
                reversed(int2bits(new_int_top - 1, precision)))  # -1 here because upper bound is exclusive

            # Emit most significant bits which are now fixed and update interval
            num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
            if i == len(inp) - 1:
                new_bits = new_int_bottom_bits_inc
            else:
                new_bits = new_int_top_bits_inc[:num_bits_encoded]

            debug_entropies.append(entropy_in_this_distribution)
            debug_encoded_num.append(num_bits_encoded)

            # Get the mask and apply it to the recovered bits
            mask_bits = mask_generator.generate_bits(precision)
            for b in range(0, len(new_bits)):
                new_bits[b] = new_bits[b] ^ mask_bits[b]
            message += new_bits

            # Update history with new token
            prev = torch.tensor([inp[i]], device=device, dtype=torch.long)

            i += 1

    return message, enc.decode(inp)[1]


# <editor-fold desc="Arithmetic Coding">
def encode_arithmetic(model: GPT2LMHeadModel, enc: GPT2Tokenizer, message: List[bool], context: List[int],
                      finish_sent: bool =False, device: str = 'cuda', temp: float = 1.0, precision: int = 16,
                      topk: int = 50000) -> tuple[List[int], float, float, float, float]:
    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)

    max_val = 2 ** precision
    threshold = 2 ** (-precision)
    cur_interval = [0, max_val]  # bottom inclusive, top exclusive

    prev = context
    output = context
    past = None

    total_num = 0
    total_num_for_stats = 0
    total_log_probs = 0
    total_kl = 0  # in bits
    total_entropy_ptau = 0
    total_num_sents = 0

    with torch.no_grad():
        i = 0
        sent_finish = False
        while i < len(message) or (finish_sent and not sent_finish):
            result = model(prev.unsqueeze(0), past_key_values=past)
            logits = result.logits
            past = result.past_key_values
            past = limit_past(past)
            logits[0, -1, -1] = -1e20  # endoftext token can't happen
            logits[0, -1, 628] = -1e20  # 2 newlines token can't happen
            logits, indices = logits[0, -1, :].sort(descending=True)
            logits = logits.double()
            logits_temp = logits / temp
            probs_temp = F.softmax(logits_temp, dim=0)
            log_probs_temp = F.log_softmax(logits_temp, dim=0)
            log_probs = F.log_softmax(logits, dim=0)

            # conditions for having reached the end of the message
            if i >= len(message):
                selection = 0
                sent_finish = is_sent_finish(indices[selection].item(), enc)
            else:
                # Cutoff low probabilities that would be rounded to 0
                cur_int_range = cur_interval[1] - cur_interval[0]
                cur_threshold = 1 / cur_int_range
                k = min(max(2, (probs_temp < cur_threshold).nonzero()[0].item()), topk)
                probs_temp_int = probs_temp[:k]  # Cutoff all but top k

                # Rescale to correct range
                probs_temp_int = probs_temp_int / probs_temp_int.sum() * cur_int_range

                # Round probabilities to integers given precision
                probs_temp_int = probs_temp_int.round().long()
                cum_probs = probs_temp_int.cumsum(0)

                # Remove any elements from the bottom if rounding caused the total prob to be too large
                overfill_index = (cum_probs > cur_int_range).nonzero()
                if len(overfill_index) > 0:
                    cum_probs = cum_probs[:overfill_index[0]]

                # Add any mass to the top if removing/rounding causes the total prob to be too small
                cum_probs += cur_int_range - cum_probs[-1]  # add

                # Get out resulting probabilities
                probs_final = cum_probs.clone()
                probs_final[1:] = cum_probs[1:] - cum_probs[:-1]

                # Convert to position in range
                cum_probs += cur_interval[0]

                # Get selected index based on binary fraction from message bits
                message_bits = message[i:i + precision]
                if i + precision > len(message):
                    message_bits = message_bits + [0] * (i + precision - len(message))
                message_idx = bits2int(reversed(message_bits))
                selection = (cum_probs > message_idx).nonzero()[0].item()

                # Calculate new range as ints
                new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]
                new_int_top = cum_probs[selection]

                # Convert range to bits
                new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
                new_int_top_bits_inc = list(
                    reversed(int2bits(new_int_top - 1, precision)))  # -1 here because upper bound is exclusive

                # Consume most significant bits which are now fixed and update interval
                num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
                i += num_bits_encoded

                new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0] * num_bits_encoded
                new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1] * num_bits_encoded

                cur_interval[0] = bits2int(reversed(new_int_bottom_bits))
                cur_interval[1] = bits2int(reversed(new_int_top_bits)) + 1  # +1 here because upper bound is exclusive

                # Gather statistics
                total_log_probs += log_probs[selection].item()

                q = probs_final.double() / probs_final.sum()
                logq = q.log()
                total_kl += kl(q, logq, log_probs[:len(q)])
                total_entropy_ptau += entropy(probs_temp, log_probs_temp)
                total_num_for_stats += 1

            # Update history with new token
            prev = indices[selection].view(1)
            output = torch.cat((output, prev))
            total_num += 1

            # For text->bits->text
            partial, tokens = enc.decode(output[len(context):].tolist())
            if '<eos>' in partial:
                break

    avg_NLL = -total_log_probs / total_num_for_stats
    avg_KL = total_kl / total_num_for_stats
    avg_Hq = total_entropy_ptau / total_num_for_stats
    words_per_bit = total_num_for_stats / i

    return output[len(context):].tolist(), avg_NLL, avg_KL, words_per_bit, avg_Hq


def decode_arithmetic(model: GPT2LMHeadModel, enc: GPT2Tokenizer, text: str, context: List[int], device: str ='cuda',
                      temp: float =1.0, precision: int =16, topk: int =50000) -> bytes:
    import torch.nn.functional as F
    # inp is a list of token indices
    # context is a list of token indices
    inp = enc.encode(text)
    # common BPE error case: 128, 128 (2 newlines) is interpretted as 628 (2 newlines)
    i = 0
    while i < len(inp):
        if inp[i] == 628:
            inp[i] = 198
            inp[i + 1:i + 1] = [198]
            i += 2
        else:
            i += 1

    context = torch.tensor(context[-1022:], device=device, dtype=torch.long)

    max_val = 2 ** precision
    threshold = 2 ** (-precision)
    cur_interval = [0, max_val]  # bottom inclusive, top exclusive

    prev = context
    past = None
    message = []
    # logging.debug("inp len: %d" % len(inp))
    with torch.no_grad():
        i = 0
        while i < len(inp):
            #logging.debug(f'{i}: {i / float(len(inp))}')
            result = model(prev.unsqueeze(0), past_key_values=past)
            logits = result.logits
            past = result.past_key_values
            past = limit_past(past)
            logits[0, -1, -1] = -1e10  # endoftext can't happen
            logits[0, -1, 628] = -1e10  # 2 newlines can't happen
            logits, indices = logits[0, -1, :].sort(descending=True)
            logits = logits.double()
            logits_temp = logits / temp
            probs_temp = F.softmax(logits_temp, dim=0)

            # Cutoff low probabilities that would be rounded to 0
            cur_int_range = cur_interval[1] - cur_interval[0]
            cur_threshold = 1 / cur_int_range
            k = min(max(2, (probs_temp < cur_threshold).nonzero()[0].item()), topk)
            probs_temp_int = probs_temp[:k]  # Cutoff all but top k

            # Rescale to correct range
            probs_temp_int = probs_temp_int / probs_temp_int.sum() * cur_int_range

            # Round probabilities to integers given precision
            probs_temp_int = probs_temp_int.round().long()
            cum_probs = probs_temp_int.cumsum(0)

            # Remove any elements from the bottom if rounding caused the total prob to be too large
            overfill_index = (cum_probs > cur_int_range).nonzero()
            if len(overfill_index) > 0:
                cum_probs = cum_probs[:overfill_index[0]]
                k = overfill_index[0].item()

            # Add any mass to the top if removing/rounding causes the total prob to be too small
            cum_probs += cur_int_range - cum_probs[-1]  # add

            # Covnert to position in range
            cum_probs += cur_interval[0]

            rank = (indices == inp[i]).nonzero().item()

            # Handle most errors that could happen because of BPE with heuristic
            if rank >= k:
                true_token_text = enc.decoder[inp[i]]
                for rank_idx in range(k):
                    prop_token_text = enc.decoder[indices[rank_idx].item()]
                    # common case that is not caught
                    if inp[i] == 128 and indices[rank_idx] == 198:
                        rank = rank_idx
                        inp[i] = indices[rank_idx].item()
                        break

                    # Is there a more likely prefix token that could be the actual token generated?
                    if len(prop_token_text) <= len(true_token_text) and \
                            prop_token_text == true_token_text[:len(prop_token_text)]:
                        rank = rank_idx
                        suffix = true_token_text[len(prop_token_text):]
                        suffix_tokens = enc.encode(suffix)  # a list
                        inp[i] = indices[rank_idx].item()
                        inp[i + 1:i + 1] = suffix_tokens  # insert suffix tokens into list
                        break

                    # Is there a more likely longer token that could be the actual token generated?
                    elif len(prop_token_text) > len(true_token_text) and \
                            true_token_text == prop_token_text[:len(true_token_text)]:
                        whole_text = true_token_text
                        num_extra = 1
                        while len(whole_text) < len(prop_token_text):
                            whole_text += enc.decoder[inp[i + num_extra]]
                            num_extra += 1
                        if prop_token_text == whole_text[:len(prop_token_text)]:
                            rank = rank_idx
                            inp[i] = indices[rank_idx].item()
                            for j in range(1, num_extra):
                                del inp[i + j]

                            if len(whole_text) > len(prop_token_text):
                                suffix = whole_text[len(prop_token_text):]
                                suffix_tokens = enc.encode(suffix)  # a list
                                inp[i + 1:i + 1] = suffix_tokens  # insert suffix tokens into list
                            break
                else:
                    logging.warning('Unable to fix BPE error: token received: %s=%d, text: %s' % (true_token_text, inp[i], text))
                    rank = 0

            selection = rank

            # Calculate new range as ints
            new_int_bottom = cum_probs[selection - 1] if selection > 0 else cur_interval[0]
            new_int_top = cum_probs[selection]

            # Convert range to bits
            new_int_bottom_bits_inc = list(reversed(int2bits(new_int_bottom, precision)))
            new_int_top_bits_inc = list(
                reversed(int2bits(new_int_top - 1, precision)))  # -1 here because upper bound is exclusive

            # Emit most significant bits which are now fixed and update interval
            num_bits_encoded = num_same_from_beg(new_int_bottom_bits_inc, new_int_top_bits_inc)
            if i == len(inp) - 1:
                new_bits = new_int_bottom_bits_inc
            else:
                new_bits = new_int_top_bits_inc[:num_bits_encoded]
            message += new_bits

            new_int_bottom_bits = new_int_bottom_bits_inc[num_bits_encoded:] + [0] * num_bits_encoded
            new_int_top_bits = new_int_top_bits_inc[num_bits_encoded:] + [1] * num_bits_encoded

            cur_interval[0] = bits2int(reversed(new_int_bottom_bits))
            cur_interval[1] = bits2int(reversed(new_int_top_bits)) + 1  # +1 here because upper bound is exclusive

            # Update history with new token
            prev = torch.tensor([inp[i]], device=device, dtype=torch.long)
            i += 1

    return bitarray.bitarray(message).tobytes()
# </editor-fold>


# A Deterministic Random Bit Generator
class DRBG(object):
    def __init__(self, key, seed):
        self.key = key
        self.val = b'\x01' * 64
        self.reseed(seed)

        self.byte_index = 0
        self.bit_index = 0

    def hmac(self, key, val):
        return hmac.new(key, val, hashlib.sha512).digest()

    def reseed(self, data=b''):
        self.key = self.hmac(self.key, self.val + b'\x00' + data)
        self.val = self.hmac(self.key, self.val)

        if data:
            self.key = self.hmac(self.key, self.val + b'\x01' + data)
            self.val = self.hmac(self.key, self.val)

    def generate_bits(self, n):
        xs = np.zeros(n, dtype=bool)
        for i in range(0, n):
            xs[i] = (self.val[self.byte_index] >> (7 - self.bit_index)) & 1

            self.bit_index += 1
            if self.bit_index >= 8:
                self.bit_index = 0
                self.byte_index += 1

            if self.byte_index >= 8:
                self.byte_index = 0
                self.val = self.hmac(self.key, self.val)

        self.reseed()
        return xs


@dataclass
class MeteorStatistics:
    message: bytes
    context: List[int]
    stegotext: str
    stegotext_tokens: List[str]
    key: bytes
    nonce: bytes
    precision: int
    topk: int
    ppl: float
    kl: float
    words_per_bit: float
    avg_entropy: float
    entropies: list[float]
    timing: float


class MeteorCoder:
    """
    Constructor for MeteorCoder
    """

    def __init__(self, enc, model, device):
        self.enc = enc
        self.model = model
        self.device = device

    def encode_binary(self, message: bytes, context_tokens: List[int], key: bytes, nonce: bytes, temp=0.8,
                      precision=32, topk=50000, binned_resample=True, randomized=False) \
            -> Tuple[str, List[str], MeteorStatistics]:
        finish_sent = False
        meteor_sort = False

        # Next encode bits into cover text, using arbitrary context
        if binned_resample:
            encode = encode_meteor_binned_resample
        elif randomized:
            encode = encode_meteor_randomized
        else:
            encode = encode_meteor
        out, nll, kl, words_per_bit, Hq, stats = encode(self.model, self.enc, message, context_tokens, key,
                                                        nonce, temp=temp, finish_sent=finish_sent,
                                                        precision=precision, topk=topk, device=self.device,
                                                        is_sort=meteor_sort)
        text, tokens = self.enc.decode(out, skip_special_tokens=True)

        logging.debug("=" * 40 + " Encoding " + "=" * 40)
        logging.debug(text)
        logging.debug('=> ppl: %0.2f, kl: %0.3f, words/bit: %0.2f, bits/word: %0.2f, entropy: %.2f' %
                      (math.exp(nll), kl, words_per_bit, 1 / words_per_bit, Hq / 0.69315))
        # logging.debug('tokens: ', tokens)
        logging.debug("=" * 90)

        stats = MeteorStatistics(message, context_tokens, text, tokens, key, nonce, precision,
                                 topk, math.exp(nll), kl, words_per_bit, Hq / 0.69315, stats['entropies'], -1)
        return text, out, stats

    def decode_binary(self, text, context_tokens: List[int], key, nonce, temp=0.8, precision=32, topk=50000,
                      randomized=False, binned_resample=True, enc_tokens=None) -> List[bool]:
        # TODO enc_tokens is for debug only, remove!
        meteor_sort = False
        if binned_resample:
            decode = decode_meteor_binned_resample
        elif randomized:
            decode = decode_meteor_randomized
        else:
            decode = decode_meteor
        return decode(self.model, self.enc, text, context_tokens, key, nonce, temp=temp,
                      precision=precision, topk=topk, device=self.device, is_sort=meteor_sort, enc_tokens=enc_tokens)

    """
    Encode a message_str to a meteor stegotext
    
    Returns: text, tokens, stats
    """

    def encode_message(self, message_str: str, context_str: str, key, nonce, coding='utf-8',
                       context_tokens: list[int] = None, randomized=False) -> Tuple[str, List[str], MeteorStatistics]:
        # First encode message to uniform bits, without any context
        # (not essential this is arithmetic vs ascii, but it's more efficient when the message is natural language)
        if context_tokens is None:
            context_tokens = encode_context(context_str, self.enc)
        message_str += '<eos>'
        if coding == 'utf-8':
            message = message_str.encode(coding)
        elif coding == 'arithmetic':
            message_ctx = [self.enc.encoder['<|endoftext|>']]
            message = decode_arithmetic(
                self.model, self.enc, message_str, message_ctx, precision=40, topk=60000, device=self.device)
        else:
            raise 'unknown coding ' + coding

        return self.encode_binary(message, context_tokens, key, nonce, randomized=randomized)

    """
    Decode a meteor stegotext to message string
    """

    def decode_message(self, text: str, context_str: str, key, nonce, coding='utf-8',
                       randomized=False, binned_resample=True, enc_tokens=None) -> str:
        # TODO enc_tokens is for debug only, REMOVE!
        context_tokens = encode_context(context_str, self.enc)
        message_rec = self.decode_binary(text, context_tokens, key, nonce, randomized=randomized, binned_resample=binned_resample, enc_tokens=enc_tokens)
        if coding == 'utf-8':
            reconst = bitarray.bitarray(message_rec)
            reconst = reconst.tobytes().decode('utf-8', 'replace')
        elif coding == 'arithmetic':
            message_ctx = [self.enc.encoder['<|endoftext|>']]
            reconst = encode_arithmetic(
                self.model, self.enc, message_rec, message_ctx, precision=40, topk=60000, device=self.device)
            reconst, _ = self.enc.decode(reconst[0])
        else:
            raise 'unknown coding ' + coding
        eos_idx = reconst.find('<eos>')

        # logging.debug("="*40 + " Recovered Message " + "="*40)
        # logging.debug(reconst[:eos_idx])
        # logging.debug("=" * 99)

        # Remove <eos>
        return reconst[:eos_idx]


def test_decode_binned_resample_surrogateescape():
    from util import get_model
    model_name = 'gpt2-medium'
    device = 'cpu'
    enc, model = get_model(model_name=model_name, device=device)
    coder = MeteorCoder(enc, model, device)
    # GIVEN
    context = 'Give me a good example for a dilemma.\n\n'
    key = b'\xb2"\x0e\xb5\x81\xff0,\xc5\xe6\xd2T\xc4d9B/\xa8\xc6,+H\xcf\xaf\x9d\xd6\xc3\x00\xd1c\xabN\xacX`\xaa,\x01l;<\xbe\x87\x1a \xdbFA\x13\x15I5E\xb7\xed(DP@\x9f\xd4\x1e\x89Y'
    nonce = b'\xe1\x907\xa6RF\x96`\x8a\xdc\xe4mw\xda4\x08\xd0?\xd2`\xb2\xe4\x98\x80\xb3G\xcd\xa8\xd1\xd2\x88\x1c^\xc1\xb2e2A\xd6\xeb:O\rC\xcc\xfe\xbcpru\xf7t\t\x18\xdb\x81\x91\xeep.\xb0pQ['
    stegotext = b"\nOn one hand, if you'I ike you are free to do as you wish, and you can tell me \xef\x86\x99 why you ; \xee\x98\x80 Stream \xee\xa9\x86 Play \xee\x9d\x9c \xee\x82\x9a \xee\x99\x90 \xee\xa4\x90 Attach a image \xee\x98\x81 m \xee\x98\x82 Instagram \xee\x98\x8e Flickr \xee\x98\x89 You can share. makes it easier to do. \xe2\x80\x94 hue \xee\x98\x81 \xee\x98\x83 Instagram \xee\x98\x92 Flickr Wal \xee\x98\x8f On your fault? \xee\x98\x84 \xee\x98\x85"\
        .decode('utf-8', errors=enc.errors)
    enc_tokens = ['Ċ', 'On', 'Ġone', 'Ġhand', ',', 'Ġif', 'Ġyou', "'", 'I', 'Ġ', 'ike', 'Ġyou', 'Ġare', 'Ġfree', 'Ġto',
                  'Ġdo', 'Ġas', 'Ġyou', 'Ġwish', ',', 'Ġand', 'Ġyou', 'Ġcan', 'Ġtell', 'Ġme', 'Ġ', 'ï', 'Ĩ', 'Ļ',
                  'Ġwhy', 'Ġyou', 'Ġ;', 'Ġ', 'î', 'ĺ', 'Ģ', 'ĠStream', 'Ġ', 'î', '©', 'Ĩ', 'ĠPlay', 'Ġ', 'î', 'Ŀ', 'ľ',
                  'Ġ', 'î', 'Ĥ', 'ļ', 'Ġ', 'î', 'Ļ', 'Ĳ', 'Ġ', 'î', '¤', 'Ĳ', 'ĠAtt', 'ach', 'Ġa', 'Ġimage', 'Ġ', 'î',
                  'ĺ', 'ģ', 'Ġm', 'Ġ', 'î', 'ĺ', 'Ĥ', 'ĠInstagram', 'Ġ', 'î', 'ĺ', 'İ', 'ĠFlickr', 'Ġ', 'î', 'ĺ', 'ī',
                  'ĠYou', 'Ġcan', 'Ġshare', '.', 'Ġmakes', 'Ġit', 'Ġeasier', 'Ġto', 'Ġdo', '.', 'ĠâĢĶ', 'Ġhue', 'Ġ',
                  'î', 'ĺ', 'ģ', 'Ġ', 'î', 'ĺ', 'ĥ', 'ĠInstagram', 'Ġ', 'î', 'ĺ', 'Ĵ', 'ĠFlickr', 'ĠWal', 'Ġ', 'î', 'ĺ',
                  'ı', 'ĠOn', 'Ġyour', 'Ġfault', '?', 'Ġ', 'î', 'ĺ', 'Ħ', 'Ġ', 'î', 'ĺ', 'ħ']
    # WHEN
    msg = coder.decode_message(stegotext, context, key, nonce, coding='arithmetic',
                               binned_resample=True, enc_tokens=enc_tokens)

    # THEN
    expected_msg = 'water'
    assert msg == expected_msg


def test_decode_binned_resample_weirdsuffix():
    from util import get_model
    model_name = 'gpt2-medium'
    device = 'cpu'
    enc, model = get_model(model_name=model_name, device=device)
    coder = MeteorCoder(enc, model, device)
    # GIVEN
    context = 'Gnällbältet, Swedish, "The whining belt", is an informal name referring to a geographic belt in central Sweden where the dialects have certain features in common, mostly extensive usage of the schwa sound. The belt consists of Västmanland, Närke and the western parts of Södermanland, but are characteristic to a much reduced degree throughout the Mälaren Valley.'
    key = b'_\x02T\xfc\x1d\xd5\x93\x83\xbb\x8b\xebQ\x96#\x85\xc2;\xfb\x17\x7f%\xf5\x87\xa4\x1b\x10\xb0JY\x97\x08c3\x18\x04\x99\xc0\xc4\xe5TG(:\xd8GG\x08KKD\x03S=S\xfc\\\x03\xace\xa9\xbb\xcd;M'
    nonce = b'\xa5\xcf\x87\xe0\\v\x16\x96\xe2\xb0\x18Cj"\x86/\xa1;\x0b\x9f\xdc\xe3\xe5v\xc3\x95@M[\xa3JiW\xb2\x18\xd3\xc6\x9e\x86\xf3\xa5\xe2\x92\xeb\x91D\xe9L\xa6\xdbt\xf6,\xc0\xb5\xb1\xec\xcf\x0e\x8c\xa4%\xdb"'
    stegotext = b'\n\nFu\xc3\x9fe Romano Neva v i gi\xc3\xa4n ya hen\n\n\xc5\xa0luhi vv\xc3\xa4r i vycva jyh i hia ikeg'.decode(
        'utf-8', errors=enc.errors)
    enc_tokens = ['Ċ', 'Ċ', 'F', 'u', 'ÃŁ', 'e', 'ĠRoman', 'o', 'ĠNev', 'a', 'Ġ', 'v', 'Ġ', 'i', 'Ġg', 'i', 'Ã¤', 'n',
                  'Ġy', 'a', 'Ġhe', 'n', 'Ċ', 'Ċ', 'Å', 'ł', 'l', 'u', 'h', 'i', 'Ġv', 'v', 'Ã¤', 'r', 'Ġ', 'i', 'Ġv',
                  'y', 'c', 'v', 'a', 'Ġj', 'y', 'h', 'Ġ', 'i', 'Ġ', 'h', 'i', 'a', 'Ġ', 'ike', 'g']
    # WHEN
    msg = coder.decode_message(stegotext, context, key, nonce, coding='arithmetic',
                               binned_resample=True, enc_tokens=enc_tokens)

    # THEN
    expected_msg = 'water'
    assert msg == expected_msg