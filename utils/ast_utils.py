from tree_sitter_languages import get_parser
import os
import json
from utils.log import write_to_log

# Critical control flow nodes (for defect detection)
CRITICAL_CONTROL_FLOW = {
    "if_statement",
    "for_statement",
    "while_statement",
    "do_statement",
    "switch_statement",
}

# High-risk operation nodes
HIGH_RISK_OPERATIONS = {
    "call_expression",          # Function calls (potential null/error)
    "assignment_expression",    # Assignments (potential overwrites)
    "return_statement",         # Returns (logic flow)
    "break_statement",          # Early exits
    "continue_statement",       # Loop control
    "goto_statement",           # Dangerous control flow
}

# Nodes to collect for statistics only
STATS_ONLY_NODES = {
    "binary_expression",
    "unary_expression",
    "pointer_expression",
    "case_statement",
}

# Use pre-compiled language libraries for AST parsing
parser = get_parser('c')  # For simplicity, we treat it as C code

def traverse_for_features(node, features):
    node_type = node.type

    # Collect critical control flow nodes (preserve order)
    if node_type in CRITICAL_CONTROL_FLOW:
        features['control_flow'].append(node_type)

    # Collect high-risk operations (preserve order, but limit count)
    if node_type in HIGH_RISK_OPERATIONS:
        features['operations'].append(node_type)

    # Update statistics
    if node_type in STATS_ONLY_NODES:
        features['stats'][node_type] = features['stats'].get(node_type, 0) + 1

    # Recursively traverse children
    for child in node.children:
        traverse_for_features(child, features)


def extract_ast_features(code: str, max_control_flow: int = 20, max_operations: int = 30):
    try:
        tree = parser.parse(code.encode("utf8"))
        root = tree.root_node

        features = {
            'control_flow': [],
            'operations': [],
            'stats': {}
        }

        traverse_for_features(root, features)

        # Truncate to prevent sequences from being too long
        features['control_flow'] = features['control_flow'][:max_control_flow]
        features['operations'] = features['operations'][:max_operations]

        return features
    except Exception as e:
        # If parsing fails, return empty features
        return {
            'control_flow': [],
            'operations': [],
            'stats': {}
        }


def make_augmented_text(code: str, max_control_flow: int = 20, max_operations: int = 30):
    """
    Create augmented text with lightweight AST features.
    Format: [CODE] + code + [AST-CF] + control_flow + [AST-OP] + operations + [AST-STATS] + stats
    """
    features = extract_ast_features(code, max_control_flow, max_operations)

    # Build compact AST representation
    parts = [f"[CODE]\n{code}"]

    # Add control flow sequence 
    if features['control_flow']:
        cf_str = " ".join(features['control_flow'])
        parts.append(f"\n[AST-CF] {cf_str}")

    # Add operations sequence
    if features['operations']:
        op_str = " ".join(features['operations'])
        parts.append(f"\n[AST-OP] {op_str}")

    # Add statistics as compact key:value pairs
    if features['stats']:
        stats_items = [f"{k}:{v}" for k, v in sorted(features['stats'].items())]
        stats_str = " ".join(stats_items)
        parts.append(f"\n[AST-STATS] {stats_str}")

    return "".join(parts)
