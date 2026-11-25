from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from tree_sitter_language_pack import get_parser


EXTENSION_TO_LANG = {
    ".py": "python",
    ".java": "java",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
}

TARGET_NODE_TYPES = {
    "python": {"function_definition", "class_definition"},
    "java": {"method_declaration", "class_declaration", "interface_declaration"},
    "javascript": {"function_declaration", "class_declaration", "method_definition"},
    "typescript": {"function_declaration", "class_declaration", "method_definition"},
    "tsx": {"function_declaration", "class_declaration", "method_definition"},
}


@dataclass
class CodeChunk:
    file_path: str
    language: str
    node_type: str
    symbol_name: str
    start_line: int
    end_line: int
    code: str
    calls: List[str] = field(default_factory=list)


class ASTCodeParser:
    """Parses source files into semantically intact functions/classes using Tree-sitter."""

    def __init__(self) -> None:
        self._parsers = {}

    def _get_lang_parser(self, language: str):
        if language not in self._parsers:
            self._parsers[language] = get_parser(language)
        return self._parsers[language]

    def _extract_identifier(self, node, source_bytes: bytes) -> str:
        for child in node.children:
            if child.type in {"identifier", "type_identifier", "property_identifier"}:
                return source_bytes[child.start_byte : child.end_byte].decode("utf-8", errors="replace")
        return "anonymous"

    def _extract_calls(self, node, source_bytes: bytes) -> List[str]:
        calls = set()

        def walk(n):
            if n.type in {"call", "call_expression", "method_invocation"}:
                func_node = n.child_by_field_name("function") or n.child_by_field_name("name")
                if func_node:
                    raw = source_bytes[func_node.start_byte : func_node.end_byte].decode("utf-8", errors="replace")
                    calls.add(raw.split(".")[-1].strip())
            for child in n.children:
                walk(child)

        walk(node)
        return sorted(calls)

    def parse_file(self, file_path: Path, repo_root: Optional[Path] = None) -> List[CodeChunk]:
        language = EXTENSION_TO_LANG.get(file_path.suffix.lower())
        if not language:
            return []

        source_bytes = file_path.read_bytes()
        if not source_bytes.strip():
            return []

        parser = self._get_lang_parser(language)
        tree = parser.parse(source_bytes)
        target_types = TARGET_NODE_TYPES[language]
        rel_path = str(file_path.relative_to(repo_root)) if repo_root else str(file_path)

        chunks: List[CodeChunk] = []

        def traverse(node):
            if node.type in target_types:
                code_text = source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
                symbol_name = self._extract_identifier(node, source_bytes)
                calls = self._extract_calls(node, source_bytes)
                chunks.append(
                    CodeChunk(
                        file_path=rel_path,
                        language=language,
                        node_type=node.type,
                        symbol_name=symbol_name,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        code=code_text,
                        calls=calls,
                    )
                )
            for child in node.children:
                traverse(child)

        traverse(tree.root_node)
        return chunks


if __name__ == "__main__":
    parser = ASTCodeParser()
    extracted = parser.parse_file(Path(__file__))
    for c in extracted:
        print(f"[{c.node_type}] {c.symbol_name} (lines {c.start_line}-{c.end_line}) -> calls: {c.calls}")
