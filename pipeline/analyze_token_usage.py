#!/usr/bin/env python3
"""
Analyze token usage from Codex trace JSONL files.

Shows:
- Token usage breakdown by type (input, output, cached, reasoning)
- Token usage per turn
- Cumulative usage over time
- Token usage between turns
- Generates graphs visualizing token usage
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Graphs will not be generated.", file=sys.stderr)

def parse_trace_file(trace_file: Path) -> Dict:
    """Parse the trace JSONL file and extract token usage data."""
    turns = []
    token_events = []
    current_turn = None
    turn_num = 0
    
    # Track activities that contribute to input tokens
    file_reads = []
    function_calls = []
    agent_messages = []
    call_id_to_function = {}  # Map call_id to function name
    
    with open(trace_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                event_type = event.get('type', '')
                payload = event.get('payload', {})
                
                # Track turn context changes (new turn starts)
                if event_type == 'turn_context':
                    if current_turn is None or current_turn.get('cwd') != payload.get('cwd'):
                        turn_num += 1
                        current_turn = {
                            'turn_num': turn_num,
                            'timestamp': event.get('timestamp', ''),
                            'cwd': payload.get('cwd', ''),
                        }
                        turns.append(current_turn)
                
                # Track activities that contribute to input tokens
                if event_type == 'response_item':
                    item_type = payload.get('type', '')
                    if item_type == 'function_call':
                        call_id = payload.get('call_id', '')
                        func_name = payload.get('name', '')
                        call_id_to_function[call_id] = func_name
                        function_calls.append({
                            'turn_num': turn_num if current_turn else 0,
                            'name': func_name,
                            'arguments': payload.get('arguments', ''),
                            'call_id': call_id,
                            'timestamp': event.get('timestamp', ''),
                        })
                    elif item_type == 'function_call_output':
                        call_id = payload.get('call_id', '')
                        func_name = call_id_to_function.get(call_id, '')
                        output = payload.get('output', '')
                        output_str = str(output)
                        
                        # Track file reads
                        if func_name == 'read_file':
                            file_reads.append({
                                'turn_num': turn_num if current_turn else 0,
                                'output_length': len(output_str),
                                'call_id': call_id,
                                'timestamp': event.get('timestamp', ''),
                            })
                    elif item_type == 'message' and payload.get('role') == 'assistant':
                        text = ''
                        for content in payload.get('content', []):
                            if content.get('type') == 'output_text':
                                text += content.get('text', '')
                        if text:
                            agent_messages.append({
                                'turn_num': turn_num if current_turn else 0,
                                'length': len(text),
                                'timestamp': event.get('timestamp', ''),
                            })
                
                # Extract token count events
                if event_type == 'event_msg' and payload.get('type') == 'token_count':
                    info = payload.get('info')
                    if info:
                        total_usage = info.get('total_token_usage', {})
                        last_usage = info.get('last_token_usage', {})
                        
                        token_event = {
                            'turn_num': turn_num if current_turn else 0,
                            'timestamp': event.get('timestamp', ''),
                            'total': {
                                'input_tokens': total_usage.get('input_tokens', 0),
                                'output_tokens': total_usage.get('output_tokens', 0),
                                'cached_input_tokens': total_usage.get('cached_input_tokens', 0),
                                'reasoning_output_tokens': total_usage.get('reasoning_output_tokens', 0),
                                'total_tokens': total_usage.get('total_tokens', 0),
                            },
                            'last': {
                                'input_tokens': last_usage.get('input_tokens', 0),
                                'output_tokens': last_usage.get('output_tokens', 0),
                                'cached_input_tokens': last_usage.get('cached_input_tokens', 0),
                                'reasoning_output_tokens': last_usage.get('reasoning_output_tokens', 0),
                                'total_tokens': last_usage.get('total_tokens', 0),
                            }
                        }
                        token_events.append(token_event)
                        
                        # Update current turn with latest usage
                        if current_turn:
                            current_turn.update({
                                'input_tokens': token_event['total']['input_tokens'],
                                'output_tokens': token_event['total']['output_tokens'],
                                'cached_input_tokens': token_event['total']['cached_input_tokens'],
                                'reasoning_output_tokens': token_event['total']['reasoning_output_tokens'],
                                'total_tokens': token_event['total']['total_tokens'],
                                'last_input_tokens': token_event['last']['input_tokens'],
                                'last_output_tokens': token_event['last']['output_tokens'],
                                'last_cached_input_tokens': token_event['last']['cached_input_tokens'],
                                'last_reasoning_output_tokens': token_event['last']['reasoning_output_tokens'],
                                'last_total_tokens': token_event['last']['total_tokens'],
                            })
                            
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}", file=sys.stderr)
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}", file=sys.stderr)
                continue
    
    return {
        'turns': turns,
        'token_events': token_events,
        'file_reads': file_reads,
        'function_calls': function_calls,
        'agent_messages': agent_messages,
    }

def calculate_turn_deltas(turns: List[Dict]) -> List[Dict]:
    """Calculate token usage deltas between turns."""
    turn_deltas = []
    
    for i, turn in enumerate(turns):
        if i == 0:
            # First turn - use its own usage
            delta = {
                'turn_num': turn['turn_num'],
                'input_tokens': turn.get('input_tokens', 0),
                'output_tokens': turn.get('output_tokens', 0),
                'cached_input_tokens': turn.get('cached_input_tokens', 0),
                'reasoning_output_tokens': turn.get('reasoning_output_tokens', 0),
                'total_tokens': turn.get('total_tokens', 0),
            }
        else:
            # Calculate delta from previous turn
            prev_turn = turns[i-1]
            delta = {
                'turn_num': turn['turn_num'],
                'input_tokens': turn.get('input_tokens', 0) - prev_turn.get('input_tokens', 0),
                'output_tokens': turn.get('output_tokens', 0) - prev_turn.get('output_tokens', 0),
                'cached_input_tokens': turn.get('cached_input_tokens', 0) - prev_turn.get('cached_input_tokens', 0),
                'reasoning_output_tokens': turn.get('reasoning_output_tokens', 0) - prev_turn.get('reasoning_output_tokens', 0),
                'total_tokens': turn.get('total_tokens', 0) - prev_turn.get('total_tokens', 0),
            }
        turn_deltas.append(delta)
    
    return turn_deltas

def print_summary(data: Dict):
    """Print token usage summary."""
    turns = data['turns']
    token_events = data['token_events']
    file_reads = data.get('file_reads', [])
    function_calls = data.get('function_calls', [])
    agent_messages = data.get('agent_messages', [])
    
    if not turns:
        print("No turns found in trace file.")
        return
    
    # Get final totals
    final_turn = turns[-1]
    total_input = final_turn.get('input_tokens', 0)
    total_output = final_turn.get('output_tokens', 0)
    total_cached = final_turn.get('cached_input_tokens', 0)
    total_reasoning = final_turn.get('reasoning_output_tokens', 0)
    total_all = final_turn.get('total_tokens', 0)
    
    print("=" * 100)
    print("TOKEN USAGE SUMMARY")
    print("=" * 100)
    print(f"\nTotal Session Tokens: {total_all:,}")
    print(f"  Input tokens:        {total_input:,} ({total_input/total_all*100:.1f}%)")
    print(f"  Output tokens:       {total_output:,} ({total_output/total_all*100:.1f}%)")
    print(f"  Cached input tokens: {total_cached:,} ({total_cached/total_all*100:.1f}%)")
    print(f"  Reasoning tokens:    {total_reasoning:,} ({total_reasoning/total_all*100:.1f}%)")
    print(f"\nTotal turns: {len(turns)}")
    print(f"Token events: {len(token_events)}")
    
    # Analyze input token contributors
    print("\n" + "=" * 100)
    print("INPUT TOKEN CONTRIBUTORS")
    print("=" * 100)
    print(f"File read operations: {len(file_reads)}")
    if file_reads:
        total_file_size = sum(f.get('output_length', 0) for f in file_reads)
        avg_file_size = total_file_size / len(file_reads) if file_reads else 0
        print(f"  Total file content read: {total_file_size:,} characters")
        print(f"  Average file size: {avg_file_size:,.0f} characters")
        print(f"  Estimated tokens (rough): ~{total_file_size // 4:,} tokens (assuming ~4 chars/token)")
    
    print(f"\nFunction calls: {len(function_calls)}")
    if function_calls:
        # Count by function type
        func_counts = defaultdict(int)
        for fc in function_calls:
            func_counts[fc.get('name', 'unknown')] += 1
        print("  Top function calls:")
        for func_name, count in sorted(func_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"    {func_name}: {count}")
    
    print(f"\nAgent messages (output): {len(agent_messages)}")
    if agent_messages:
        total_output_chars = sum(m.get('length', 0) for m in agent_messages)
        print(f"  Total output characters: {total_output_chars:,}")
        print(f"  Estimated output tokens: ~{total_output_chars // 4:,} tokens")
    
    # Calculate input vs output ratio
    if total_all > 0:
        input_ratio = (total_input + total_cached) / total_all * 100
        output_ratio = total_output / total_all * 100
        reasoning_ratio = total_reasoning / total_all * 100
        print(f"\n" + "-" * 100)
        print(f"Token Distribution:")
        print(f"  Input + Cached: {input_ratio:.1f}% ({total_input + total_cached:,} tokens)")
        print(f"    - Input tokens: {total_input:,} ({total_input/total_all*100:.1f}%)")
        print(f"    - Cached tokens: {total_cached:,} ({total_cached/total_all*100:.1f}%)")
        print(f"  Output: {output_ratio:.1f}% ({total_output:,} tokens)")
        print(f"  Reasoning: {reasoning_ratio:.1f}% ({total_reasoning:,} tokens)")
        
        if input_ratio > 90:
            print(f"\n⚠️  HIGH INPUT TOKEN USAGE ({input_ratio:.1f}%)")
            print(f"   This is common in Codex sessions because:")
            print(f"   - System prompts and instructions are included in every turn")
            print(f"   - File contents read via tools add to input tokens")
            print(f"   - Conversation history accumulates over turns")
            print(f"   - Cached tokens ({total_cached:,}) are reused context (efficient!)")
            if total_cached > 0:
                cache_efficiency = total_cached / (total_input + total_cached) * 100
                print(f"   - {cache_efficiency:.1f}% of input tokens are cached (good for efficiency)")
    
    # Calculate turn deltas
    turn_deltas = calculate_turn_deltas(turns)
    
    print("\n" + "=" * 100)
    print("TOKEN USAGE BY TURN")
    print("=" * 100)
    print(f"{'Turn':<6} {'Input':<12} {'Output':<12} {'Cached':<12} {'Reasoning':<12} {'Total':<12} {'Delta':<12}")
    print("-" * 100)
    
    for i, (turn, delta) in enumerate(zip(turns, turn_deltas)):
        turn_num = turn.get('turn_num', i+1)
        print(f"{turn_num:<6} "
              f"{turn.get('input_tokens', 0):<12,} "
              f"{turn.get('output_tokens', 0):<12,} "
              f"{turn.get('cached_input_tokens', 0):<12,} "
              f"{turn.get('reasoning_output_tokens', 0):<12,} "
              f"{turn.get('total_tokens', 0):<12,} "
              f"{delta.get('total_tokens', 0):<12,}")
    
    # Show top turns by usage
    sorted_turns = sorted(turns, key=lambda x: x.get('total_tokens', 0), reverse=True)
    
    print("\n" + "=" * 100)
    print("TOP 5 TURNS BY TOKEN USAGE")
    print("=" * 100)
    for i, turn in enumerate(sorted_turns[:5], 1):
        print(f"\n{i}. Turn {turn.get('turn_num', '?')} - {turn.get('total_tokens', 0):,} total tokens")
        print(f"   Input: {turn.get('input_tokens', 0):,} | "
              f"Output: {turn.get('output_tokens', 0):,} | "
              f"Cached: {turn.get('cached_input_tokens', 0):,} | "
              f"Reasoning: {turn.get('reasoning_output_tokens', 0):,}")
    
    # Show token usage progression
    if len(token_events) > 1:
        print("\n" + "=" * 100)
        print("TOKEN USAGE PROGRESSION (Sample of token count events)")
        print("=" * 100)
        print(f"{'Event':<8} {'Cumulative Total':<18} {'Last Turn Input':<18} {'Last Turn Output':<18} {'Last Turn Total':<18}")
        print("-" * 100)
        
        # Show every Nth event to avoid too much output
        step = max(1, len(token_events) // 20)
        for i, event in enumerate(token_events[::step]):
            print(f"{i*step+1:<8} "
                  f"{event['total']['total_tokens']:<18,} "
                  f"{event['last']['input_tokens']:<18,} "
                  f"{event['last']['output_tokens']:<18,} "
                  f"{event['last']['total_tokens']:<18,}")

def create_graphs(data: Dict, output_dir: Path):
    """Create visualization graphs for token usage."""
    if not HAS_MATPLOTLIB:
        print("\nSkipping graph generation (matplotlib not available)")
        return
    
    turns = data['turns']
    token_events = data['token_events']
    
    if not turns or not token_events:
        print("\nNo data available for graph generation")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse timestamps
    timestamps = []
    for event in token_events:
        try:
            ts = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
            timestamps.append(ts)
        except:
            timestamps.append(None)
    
    # Filter out None timestamps
    valid_indices = [i for i, ts in enumerate(timestamps) if ts is not None]
    if not valid_indices:
        print("\nNo valid timestamps found for graph generation")
        return
    
    # 1. Cumulative Token Usage Over Time
    fig, ax = plt.subplots(figsize=(12, 6))
    valid_timestamps = [timestamps[i] for i in valid_indices]
    valid_events = [token_events[i] for i in valid_indices]
    
    total_tokens = [e['total']['total_tokens'] for e in valid_events]
    input_tokens = [e['total']['input_tokens'] for e in valid_events]
    output_tokens = [e['total']['output_tokens'] for e in valid_events]
    cached_tokens = [e['total']['cached_input_tokens'] for e in valid_events]
    reasoning_tokens = [e['total']['reasoning_output_tokens'] for e in valid_events]
    
    ax.plot(valid_timestamps, total_tokens, label='Total', linewidth=2, color='black')
    ax.plot(valid_timestamps, input_tokens, label='Input', linewidth=1.5, alpha=0.7)
    ax.plot(valid_timestamps, output_tokens, label='Output', linewidth=1.5, alpha=0.7)
    ax.plot(valid_timestamps, cached_tokens, label='Cached Input', linewidth=1.5, alpha=0.7)
    ax.plot(valid_timestamps, reasoning_tokens, label='Reasoning', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative Tokens')
    ax.set_title('Cumulative Token Usage Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    graph_path = output_dir / 'cumulative_tokens_over_time.png'
    plt.savefig(graph_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {graph_path}")
    plt.close()
    
    # 2. Token Usage by Type (Pie Chart)
    if turns:
        final_turn = turns[-1]
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sizes = [
            final_turn.get('input_tokens', 0),
            final_turn.get('output_tokens', 0),
            final_turn.get('cached_input_tokens', 0),
            final_turn.get('reasoning_output_tokens', 0),
        ]
        labels = ['Input', 'Output', 'Cached Input', 'Reasoning']
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        # Filter out zero values
        non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
        if non_zero:
            sizes, labels, colors = zip(*non_zero)
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title('Token Usage Breakdown by Type')
            
            graph_path = output_dir / 'token_breakdown_pie.png'
            plt.savefig(graph_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {graph_path}")
        plt.close()
    
    # 3. Per-Turn Token Usage (Bar Chart)
    if len(turns) > 1:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        turn_nums = [t.get('turn_num', i+1) for i, t in enumerate(turns)]
        turn_totals = [t.get('total_tokens', 0) for t in turns]
        turn_inputs = [t.get('input_tokens', 0) for t in turns]
        turn_outputs = [t.get('output_tokens', 0) for t in turns]
        turn_cached = [t.get('cached_input_tokens', 0) for t in turns]
        turn_reasoning = [t.get('reasoning_output_tokens', 0) for t in turns]
        
        x = range(len(turn_nums))
        width = 0.6
        
        ax.bar(x, turn_inputs, width, label='Input', alpha=0.8, color='#ff9999')
        bottom = turn_inputs
        ax.bar(x, turn_outputs, width, bottom=bottom, label='Output', alpha=0.8, color='#66b3ff')
        bottom = [b + o for b, o in zip(bottom, turn_outputs)]
        ax.bar(x, turn_cached, width, bottom=bottom, label='Cached', alpha=0.8, color='#99ff99')
        bottom = [b + c for b, c in zip(bottom, turn_cached)]
        ax.bar(x, turn_reasoning, width, bottom=bottom, label='Reasoning', alpha=0.8, color='#ffcc99')
        
        ax.set_xlabel('Turn Number')
        ax.set_ylabel('Cumulative Tokens')
        ax.set_title('Cumulative Token Usage by Turn')
        ax.set_xticks(x)
        ax.set_xticklabels(turn_nums)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        graph_path = output_dir / 'tokens_by_turn.png'
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {graph_path}")
        plt.close()
        
        # 4. Per-Turn Token Deltas (Line Chart)
        turn_deltas = calculate_turn_deltas(turns)
        fig, ax = plt.subplots(figsize=(14, 6))
        
        delta_totals = [d.get('total_tokens', 0) for d in turn_deltas]
        delta_inputs = [d.get('input_tokens', 0) for d in turn_deltas]
        delta_outputs = [d.get('output_tokens', 0) for d in turn_deltas]
        delta_cached = [d.get('cached_input_tokens', 0) for d in turn_deltas]
        delta_reasoning = [d.get('reasoning_output_tokens', 0) for d in turn_deltas]
        
        ax.plot(turn_nums, delta_totals, marker='o', label='Total', linewidth=2, color='black')
        ax.plot(turn_nums, delta_inputs, marker='s', label='Input', linewidth=1.5, alpha=0.7)
        ax.plot(turn_nums, delta_outputs, marker='^', label='Output', linewidth=1.5, alpha=0.7)
        ax.plot(turn_nums, delta_cached, marker='v', label='Cached', linewidth=1.5, alpha=0.7)
        ax.plot(turn_nums, delta_reasoning, marker='d', label='Reasoning', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Turn Number')
        ax.set_ylabel('Token Delta (vs Previous Turn)')
        ax.set_title('Token Usage Delta Per Turn')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        graph_path = output_dir / 'token_deltas_per_turn.png'
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {graph_path}")
        plt.close()
    
    # 5. Last Turn Token Usage (for each token count event)
    if len(token_events) > 1:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        event_nums = list(range(1, len(token_events) + 1))
        last_totals = [e['last']['total_tokens'] for e in token_events]
        last_inputs = [e['last']['input_tokens'] for e in token_events]
        last_outputs = [e['last']['output_tokens'] for e in token_events]
        last_cached = [e['last']['cached_input_tokens'] for e in token_events]
        last_reasoning = [e['last']['reasoning_output_tokens'] for e in token_events]
        
        # Sample if too many events
        if len(event_nums) > 100:
            step = len(event_nums) // 100
            event_nums = event_nums[::step]
            last_totals = last_totals[::step]
            last_inputs = last_inputs[::step]
            last_outputs = last_outputs[::step]
            last_cached = last_cached[::step]
            last_reasoning = last_reasoning[::step]
        
        ax.plot(event_nums, last_totals, marker='o', markersize=3, label='Total', linewidth=1.5, color='black', alpha=0.7)
        ax.plot(event_nums, last_inputs, marker='s', markersize=2, label='Input', linewidth=1, alpha=0.6)
        ax.plot(event_nums, last_outputs, marker='^', markersize=2, label='Output', linewidth=1, alpha=0.6)
        ax.plot(event_nums, last_cached, marker='v', markersize=2, label='Cached', linewidth=1, alpha=0.6)
        ax.plot(event_nums, last_reasoning, marker='d', markersize=2, label='Reasoning', linewidth=1, alpha=0.6)
        
        ax.set_xlabel('Token Count Event Number')
        ax.set_ylabel('Last Turn Tokens')
        ax.set_title('Last Turn Token Usage Per Event')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        graph_path = output_dir / 'last_turn_tokens_per_event.png'
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {graph_path}")
        plt.close()
    
    # 6. Input Token Growth Over Time (showing what contributes)
    if len(token_events) > 1:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        valid_timestamps = [timestamps[i] for i in valid_indices]
        valid_events = [token_events[i] for i in valid_indices]
        
        input_tokens = [e['total']['input_tokens'] for e in valid_events]
        cached_tokens = [e['total']['cached_input_tokens'] for e in valid_events]
        output_tokens = [e['total']['output_tokens'] for e in valid_events]
        
        # Top plot: Input token growth
        ax1.plot(valid_timestamps, input_tokens, label='Input Tokens', linewidth=2, color='#ff6b6b')
        ax1.plot(valid_timestamps, cached_tokens, label='Cached Input Tokens', linewidth=2, color='#4ecdc4')
        ax1.fill_between(valid_timestamps, [0] * len(valid_timestamps), input_tokens, alpha=0.3, color='#ff6b6b')
        ax1.fill_between(valid_timestamps, input_tokens, [i + c for i, c in zip(input_tokens, cached_tokens)], alpha=0.3, color='#4ecdc4')
        ax1.set_ylabel('Cumulative Input Tokens', fontsize=12)
        ax1.set_title('Input Token Growth Over Time (Input + Cached)', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        
        # Bottom plot: Output vs Input ratio
        ratios = [(o / (i + c) * 100) if (i + c) > 0 else 0 
                  for i, c, o in zip(input_tokens, cached_tokens, output_tokens)]
        ax2.plot(valid_timestamps, ratios, label='Output/Input Ratio (%)', linewidth=2, color='#95e1d3')
        ax2.axhline(y=10, color='r', linestyle='--', alpha=0.5, label='10% threshold')
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Output/Input Ratio (%)', fontsize=12)
        ax2.set_title('Output Token Ratio (Output / (Input + Cached))', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        graph_path = output_dir / 'input_token_growth.png'
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {graph_path}")
        plt.close()
    
    # 7. Input Token Contributors Analysis
    file_reads = data.get('file_reads', [])
    function_calls = data.get('function_calls', [])
    
    if function_calls or file_reads:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 7a. Function calls by type (horizontal bar chart)
        if function_calls:
            func_counts = defaultdict(int)
            for fc in function_calls:
                func_counts[fc.get('name', 'unknown')] += 1
            
            # Sort by count, take top 10
            sorted_funcs = sorted(func_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            if sorted_funcs:
                func_names_sorted, func_counts_sorted = zip(*sorted_funcs)
                
                ax = axes[0]
                bars = ax.barh(range(len(func_names_sorted)), func_counts_sorted, color='steelblue', alpha=0.8)
                ax.set_yticks(range(len(func_names_sorted)))
                ax.set_yticklabels(func_names_sorted, fontsize=11)
                ax.set_xlabel('Number of Calls', fontsize=12, fontweight='bold')
                ax.set_title('Top 10 Function Calls', fontsize=13, fontweight='bold', pad=10)
                ax.grid(True, alpha=0.3, axis='x', linestyle='--')
                ax.invert_yaxis()
                
                # Add value labels
                for i, (bar, count) in enumerate(zip(bars, func_counts_sorted)):
                    width = bar.get_width()
                    ax.text(width + 0.5, i, str(count), va='center', fontsize=10, fontweight='bold')
        
        # 7b. File read statistics
        if file_reads:
            file_sizes = [f.get('output_length', 0) for f in file_reads]
            if file_sizes:
                ax = axes[1]
                
                # Create a simple bar chart showing file count and total size
                total_chars = sum(file_sizes)
                avg_size = total_chars / len(file_sizes) if file_sizes else 0
                max_size = max(file_sizes) if file_sizes else 0
                
                categories = ['Total Files', 'Total Chars', 'Avg Size', 'Max Size']
                values = [len(file_sizes), total_chars / 1000, avg_size / 1000, max_size / 1000]
                units = ['files', 'K chars', 'K chars', 'K chars']
                
                bars = ax.bar(categories, values, color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'], alpha=0.8)
                ax.set_ylabel('Value', fontsize=12, fontweight='bold')
                ax.set_title(f'File Read Statistics\n({len(file_sizes)} files read)', fontsize=13, fontweight='bold', pad=10)
                ax.grid(True, alpha=0.3, axis='y', linestyle='--')
                
                # Add value labels
                for bar, val, unit in zip(bars, values, units):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.1f}\n{unit}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 7c. Input token breakdown (stacked bar or pie)
        if turns:
            final_turn = turns[-1]
            total_input = final_turn.get('input_tokens', 0)
            total_cached = final_turn.get('cached_input_tokens', 0)
            
            # Estimate contributors
            total_file_chars = sum(f.get('output_length', 0) for f in file_reads)
            estimated_file_tokens = total_file_chars // 4
            
            # System prompts, history, etc.
            other_input = max(0, total_input - estimated_file_tokens - total_cached)
            
            ax = axes[2]
            
            # Use a simple bar chart instead of pie
            categories = []
            values = []
            colors_list = []
            
            if estimated_file_tokens > 0:
                categories.append('File Reads')
                values.append(estimated_file_tokens)
                colors_list.append('#e74c3c')
            
            if total_cached > 0:
                categories.append('Cached')
                values.append(total_cached)
                colors_list.append('#3498db')
            
            if other_input > 0:
                categories.append('Prompts/History')
                values.append(other_input)
                colors_list.append('#f39c12')
            
            if values:
                bars = ax.bar(categories, values, color=colors_list, alpha=0.8)
                ax.set_ylabel('Tokens', fontsize=12, fontweight='bold')
                ax.set_title(f'Input Token Breakdown\n(Total: {total_input:,} tokens)', fontsize=13, fontweight='bold', pad=10)
                ax.grid(True, alpha=0.3, axis='y', linestyle='--')
                
                # Add percentage labels
                total_for_pct = sum(values)
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    pct = (val / total_for_pct * 100) if total_for_pct > 0 else 0
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.suptitle('Input Token Contributors Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        graph_path = output_dir / 'input_token_contributors.png'
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {graph_path}")
        plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_token_usage.py <trace_file.jsonl> [output_dir]")
        print("  output_dir: Directory to save graphs (default: ./token_analysis_graphs)")
        sys.exit(1)
    
    trace_file = Path(sys.argv[1])
    if not trace_file.exists():
        print(f"Error: Trace file not found: {trace_file}")
        sys.exit(1)
    
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(trace_file.parent) / "token_analysis_graphs"
    
    print(f"Analyzing token usage from: {trace_file}")
    print()
    
    data = parse_trace_file(trace_file)
    print_summary(data)
    
    print("\n" + "=" * 100)
    print("GENERATING GRAPHS")
    print("=" * 100)
    create_graphs(data, output_dir)

if __name__ == '__main__':
    main()
