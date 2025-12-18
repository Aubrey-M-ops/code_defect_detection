# utils/ast_utils.py
from tree_sitter import Language, Parser
import os

# 你需要先在项目根目录编译一个共享库
# tree-sitter-c, tree-sitter-cpp
# 假设生成 build/my-languages.so
LIB_PATH = "build/my-languages.so"

if not os.path.exists(LIB_PATH):
    raise FileNotFoundError(
        "Please build tree-sitter languages and put my-languages.so in build/"
    )

Language.build_library  # 仅提醒用户存在这个 API，不一定要在这里调用

C_LANG = Language(LIB_PATH, "c")       # 和你编译时的名称一致
CPP_LANG = Language(LIB_PATH, "cpp")

parser = Parser()
parser.set_language(C_LANG)  # 简单起见，我们当成 C 处理

def traverse(node, code_bytes, tokens):
    # 收集 node type
    tokens.append(node.type)
    for child in node.children:
        traverse(child, code_bytes, tokens)

def code_to_ast_tokens(code: str, max_nodes: int = 256):
    tree = parser.parse(code.encode("utf8"))
    root = tree.root_node
    tokens = []
    traverse(root, code.encode("utf8"), tokens)
    # 为了不让 AST 太长，截断一下
    return tokens[:max_nodes]

def make_augmented_text(code: str, max_nodes: int = 256):
    ast_tokens = code_to_ast_tokens(code, max_nodes)
    ast_str = " ".join(ast_tokens)
    # 简单拼接： [AST] + [SEP] + [code]
    return f"AST_TOKENS: {ast_str}\nCODE:\n{code}"
