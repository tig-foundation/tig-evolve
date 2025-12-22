#!/usr/bin/env python3
"""
Extract and display prompts from OpenEvolve checkpoint program files in a readable format.
Usage: python view_prompt.py <program_json_file> [-o output_file.txt]
"""

import json 
import sys
import argparse
from pathlib import Path
from io import StringIO

def format_prompt(program_file):
    """Extract and format prompts from a program JSON file, return formatted string"""
    with open(program_file, 'r') as f:
        data = json.load(f)
    
    program_id = data.get('id', 'unknown')
    generation = data.get('generation', 0)
    iteration = data.get('iteration_found', 0)
    metrics = data.get('metrics', {})
    
    output = StringIO()
    
    output.write("=" * 80 + "\n")
    output.write(f"Program ID: {program_id}\n")
    output.write(f"Generation: {generation}, Iteration: {iteration}\n")
    output.write(f"Metrics: {metrics}\n")
    output.write("=" * 80 + "\n")
    output.write("\n")
    
    prompts = data.get('prompts')
    if not prompts:
        output.write("No prompts found in this program (likely generation 0 - initial program)\n")
        return output.getvalue(), program_id
    
    for template_key, prompt_data in prompts.items():
        output.write(f"\n{'='*80}\n")
        output.write(f"TEMPLATE: {template_key}\n")
        output.write(f"{'='*80}\n\n")
        
        # System message
        system = prompt_data.get('system', '')
        if system:
            output.write("SYSTEM MESSAGE:\n")
            output.write("-" * 80 + "\n")
            output.write(system + "\n")
            output.write("\n")
        
        # User message
        user = prompt_data.get('user', '')
        if user:
            output.write("USER PROMPT:\n")
            output.write("-" * 80 + "\n")
            output.write(user + "\n")
            output.write("\n")
        
        # LLM Response
        responses = prompt_data.get('responses', [])
        if responses:
            output.write("LLM RESPONSE:\n")
            output.write("-" * 80 + "\n")
            for i, response in enumerate(responses, 1):
                if len(responses) > 1:
                    output.write(f"\n[Response {i}]:\n")
                output.write(response + "\n")
            output.write("\n")
    
    return output.getvalue(), program_id

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and format prompts from OpenEvolve program JSON files')
    parser.add_argument('program_file', help='Path to the program JSON file')
    parser.add_argument('-o', '--output', help='Output file path (default: prompt_<program_id>.txt in same directory)')
    
    args = parser.parse_args()
    
    program_file = Path(args.program_file)
    if not program_file.exists():
        print(f"Error: File not found: {program_file}")
        sys.exit(1)
    
    formatted_output, program_id = format_prompt(program_file)
    
    # Determine output file path
    if args.output:
        output_file = Path(args.output)
    else:
        # Auto-generate filename based on program ID
        output_file = program_file.parent / f"prompt_{program_id}.txt"
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(formatted_output)
    
    # Also print to stdout
    print(formatted_output)
    
    print(f"\n{'='*80}")
    print(f"Saved to: {output_file}")
    print(f"{'='*80}")

