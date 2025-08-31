from flask import Flask, render_template, request
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from collections import deque
import traceback

app = Flask(__name__)

# ----------------------------
# Global Renaming Maps
# ----------------------------
global_var_map = {}
global_func_map = {}
global_var_count = 0
global_func_count = 0

def reset_global_maps():
    global global_var_map, global_func_map, global_var_count, global_func_count
    global_var_map = {}
    global_func_map = {}
    global_var_count = 0
    global_func_count = 0

# ----------------------------
# SIMPLIFIED Advanced AST Normalization
# ----------------------------
class AdvancedNormalizer(ast.NodeTransformer):
    def visit_Compare(self, node):
        # Only normalize if it's a simple comparison
        if len(node.ops) == 1 and len(node.comparators) == 1:
            if isinstance(node.ops[0], (ast.Gt, ast.GtE)):
                # Reverse comparison: a > b becomes b < a
                node.left, node.comparators[0] = node.comparators[0], node.left
                node.ops[0] = ast.Lt() if isinstance(node.ops[0], ast.Gt) else ast.LtE()
        return node
    
    def visit_BinOp(self, node):
        # Only normalize for simple expressions
        if isinstance(node.op, (ast.Add, ast.Mult)):
            left_str = ast.dump(node.left)
            right_str = ast.dump(node.right)
            if left_str > right_str:
                node.left, node.right = node.right, node.left
        return node

# ----------------------------
# Improved Canonical Order Transformer
# ----------------------------
class CanonicalOrder(ast.NodeTransformer):
    def visit_Module(self, node):
        # For simple code, don't reorder statements
        if len(node.body) <= 5:  # Small scripts don't need reordering
            return node
            
        # Separate different types of statements
        imports = []
        functions = []
        classes = []
        assigns = []
        others = []
        
        for stmt in node.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                imports.append(stmt)
            elif isinstance(stmt, ast.FunctionDef):
                functions.append(stmt)
            elif isinstance(stmt, ast.ClassDef):
                classes.append(stmt)
            elif isinstance(stmt, ast.Assign):
                assigns.append(stmt)
            else:
                others.append(stmt)
        
        # Only reorder if there are enough statements to reorder
        if len(assigns) > 3:
            ordered_assigns = self.order_assignments(assigns)
        else:
            ordered_assigns = assigns
        
        # Reconstruct body in canonical order
        node.body = imports + classes + functions + ordered_assigns + others
        return node
    
    def order_assignments(self, assigns):
        if len(assigns) <= 2:  # Don't reorder small number of assignments
            return assigns
            
        # Build dependency graph
        graph = nx.DiGraph()
        var_to_stmt = {}
        
        for stmt in assigns:
            # Get all variables defined in this assignment
            defined_vars = set()
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    defined_vars.add(target.id)
            
            # Get all variables used in this assignment
            used_vars = set()
            for node in ast.walk(stmt.value):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    used_vars.add(node.id)
            
            # Add edges for dependencies
            for defined_var in defined_vars:
                var_to_stmt[defined_var] = stmt
                for used_var in used_vars:
                    if used_var in var_to_stmt:
                        graph.add_edge(var_to_stmt[used_var], stmt)
            
            # Add the statement to the graph
            graph.add_node(stmt)
        
        # Try to get a topological order
        try:
            ordered = list(nx.topological_sort(graph))
            return ordered
        except nx.NetworkXUnfeasible:
            # If there's a cycle, use the original order
            return assigns

# ----------------------------
# Variable & Function Renaming
# ----------------------------
class RenameVariablesFunctionsGlobal(ast.NodeTransformer):
    def visit_Name(self, node):
        global global_var_count
        if isinstance(node.ctx, (ast.Store, ast.Load, ast.Del)):
            if node.id not in global_var_map:
                global_var_count += 1
                global_var_map[node.id] = f"var{global_var_count}"
            node.id = global_var_map[node.id]
        return node

    def visit_FunctionDef(self, node):
        global global_func_count
        if node.name not in global_func_map:
            global_func_count += 1
            global_func_map[node.name] = f"func{global_func_count}"
        node.name = global_func_map[node.name]
        self.generic_visit(node)
        return node
        
    def visit_ClassDef(self, node):
        # Also rename classes
        global global_func_count
        if node.name not in global_func_map:
            global_func_count += 1
            global_func_map[node.name] = f"class{global_func_count}"
        node.name = global_func_map[node.name]
        self.generic_visit(node)
        return node

# ----------------------------
# Literal Normalization
# ----------------------------
class LiteralNormalizer(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, (int, float, complex)):
            return ast.copy_location(ast.Constant(value="NUM"), node)
        elif isinstance(node.value, str):
            return ast.copy_location(ast.Constant(value="STR"), node)
        elif isinstance(node.value, bool):
            return ast.copy_location(ast.Constant(value="BOOL"), node)
        elif node.value is None:
            return ast.copy_location(ast.Constant(value="NONE"), node)
        return node

# ----------------------------
# SIMPLIFIED Structure Normalization
# ----------------------------
class StructureNormalizer(ast.NodeTransformer):
    def visit_If(self, node):
        # Only sort if there are multiple statements
        if len(node.body) > 3:
            node.body = sorted(node.body, key=lambda x: ast.dump(x))
        if node.orelse and len(node.orelse) > 3:
            node.orelse = sorted(node.orelse, key=lambda x: ast.dump(x))
        return node

# ----------------------------
# Control Flow Graph Similarity
# ----------------------------
def extract_control_flow(code):
    try:
        tree = ast.parse(code)
        cfg = {}
        
        def traverse(node, parent=None):
            node_id = id(node)
            cfg[node_id] = {
                'type': type(node).__name__,
                'children': [],
                'parent': parent
            }
            
            for child in ast.iter_child_nodes(node):
                child_id = id(child)
                cfg[node_id]['children'].append(child_id)
                traverse(child, node_id)
        
        traverse(tree)
        return cfg
    except:
        return {}

def cfg_similarity(cfg1, cfg2):
    if not cfg1 or not cfg2:
        return 0
    
    # Convert CFG to feature vectors
    type_counts1 = {}
    type_counts2 = {}
    
    for node_id, node_data in cfg1.items():
        node_type = node_data['type']
        type_counts1[node_type] = type_counts1.get(node_type, 0) + 1
    
    for node_id, node_data in cfg2.items():
        node_type = node_data['type']
        type_counts2[node_type] = type_counts2.get(node_type, 0) + 1
    
    # Calculate similarity based on node type distribution
    all_types = set(type_counts1.keys()) | set(type_counts2.keys())
    vec1 = [type_counts1.get(t, 0) for t in all_types]
    vec2 = [type_counts2.get(t, 0) for t in all_types]
    
    return cosine_similarity([vec1], [vec2])[0][0]

# ----------------------------
# Semantic Analysis
# ----------------------------
class SemanticAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.operations = []
        self.patterns = []
    
    def visit_Call(self, node):
        # Track function calls
        if isinstance(node.func, ast.Name):
            self.operations.append(f"call:{node.func.id}")
        self.generic_visit(node)
    
    def visit_For(self, node):
        # Track loop patterns
        self.patterns.append("for_loop")
        self.generic_visit(node)
    
    def visit_While(self, node):
        # Track loop patterns
        self.patterns.append("while_loop")
        self.generic_visit(node)
    
    def visit_ListComp(self, node):
        # Track comprehension patterns
        self.patterns.append("list_comprehension")
        self.generic_visit(node)

def semantic_similarity(code1, code2):
    try:
        tree1 = ast.parse(code1)
        tree2 = ast.parse(code2)
        
        analyzer1 = SemanticAnalyzer()
        analyzer1.visit(tree1)
        
        analyzer2 = SemanticAnalyzer()
        analyzer2.visit(tree2)
        
        # Compare operations
        ops1 = analyzer1.operations
        ops2 = analyzer2.operations
        
        # Compare patterns
        patterns1 = analyzer1.patterns
        patterns2 = analyzer2.patterns
        
        # Calculate similarity
        op_sim = len(set(ops1) & set(ops2)) / max(len(set(ops1)), len(set(ops2)), 1)
        pattern_sim = len(set(patterns1) & set(patterns2)) / max(len(set(patterns1)), len(set(patterns2)), 1)
        
        return (op_sim + pattern_sim) / 2
    except:
        return 0

# ----------------------------
# Code Chunking and Segment Analysis
# ----------------------------
def chunk_code(code):
    """Break code into logical chunks for more granular analysis"""
    try:
        tree = ast.parse(code)
        chunks = []
        
        class ChunkVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                chunks.append(('function', ast.unparse(node)))
                self.generic_visit(node)
            
            def visit_ClassDef(self, node):
                chunks.append(('class', ast.unparse(node)))
                self.generic_visit(node)
            
            def visit_For(self, node):
                chunks.append(('for_loop', ast.unparse(node)))
                self.generic_visit(node)
            
            def visit_While(self, node):
                chunks.append(('while_loop', ast.unparse(node)))
                self.generic_visit(node)
        
        visitor = ChunkVisitor()
        visitor.visit(tree)
        return chunks
    except:
        return [('full_code', code)]

def chunk_based_similarity(code1, code2):
    chunks1 = chunk_code(code1)
    chunks2 = chunk_code(code2)
    
    if not chunks1 or not chunks2:
        return 0
    
    # Compare each chunk type
    chunk_similarities = []
    for type1, chunk1 in chunks1:
        for type2, chunk2 in chunks2:
            if type1 == type2:
                sim = surface_similarity(chunk1, chunk2)
                chunk_similarities.append(sim)
    
    return sum(chunk_similarities) / max(len(chunk_similarities), 1) if chunk_similarities else 0

# ----------------------------
# NORMALIZATION PIPELINE WITH LEVELS
# ----------------------------
def normalize_code(code, normalization_level="auto"):
    try:
        # Try to parse the entire code first
        tree = ast.parse(code)
        
        # Auto-detect normalization level based on code size
        if normalization_level == "auto":
            code_size = len(code.split('\n'))
            if code_size <= 10:  # Small code
                normalization_level = "minimal"
            elif code_size <= 50:  # Medium code
                normalization_level = "medium"
            else:  # Large code
                normalization_level = "full"
        
        if normalization_level == "full":
            tree = CanonicalOrder().visit(tree)
            tree = AdvancedNormalizer().visit(tree)
            tree = RenameVariablesFunctionsGlobal().visit(tree)
            tree = LiteralNormalizer().visit(tree)
            tree = StructureNormalizer().visit(tree)
        elif normalization_level == "medium":
            tree = RenameVariablesFunctionsGlobal().visit(tree)
            tree = LiteralNormalizer().visit(tree)
        else:  # minimal
            tree = LiteralNormalizer().visit(tree)
        
        ast.fix_missing_locations(tree)
        normalized = ast.unparse(tree)
        
        # For identical code, ensure exact match after normalization
        if code.strip() == code.strip():  # If code is identical to itself
            # Make sure normalization doesn't change identical code
            tree2 = ast.parse(code)
            tree2 = LiteralNormalizer().visit(tree2)
            ast.fix_missing_locations(tree2)
            reference_normalized = ast.unparse(tree2)
            
            if normalized != reference_normalized:
                # Fall back to minimal normalization
                tree = ast.parse(code)
                tree = LiteralNormalizer().visit(tree)
                ast.fix_missing_locations(tree)
                normalized = ast.unparse(tree)
        
        return normalized
    except SyntaxError:
        try:
            # If the entire code fails to parse, try parsing individual statements
            parsed_statements = []
            lines = code.strip().split('\n')
            current_statement = []
            
            for line in lines:
                current_statement.append(line)
                try:
                    # Try to parse the accumulated lines
                    stmt_code = '\n'.join(current_statement)
                    stmt_tree = ast.parse(stmt_code)
                    # If successful, add to parsed statements and reset
                    parsed_statements.append(stmt_tree)
                    current_statement = []
                except SyntaxError:
                    # Continue accumulating lines
                    continue
            
            # If we have any successfully parsed statements, process them
            if parsed_statements:
                # Create a new module with all parsed statements
                new_module = ast.Module(body=[], type_ignores=[])
                for stmt in parsed_statements:
                    if isinstance(stmt, ast.Module):
                        new_module.body.extend(stmt.body)
                
                # Use minimal normalization for problematic code
                tree = LiteralNormalizer().visit(new_module)
                ast.fix_missing_locations(tree)
                return ast.unparse(tree)
            else:
                return f"Error: Could not parse any valid Python code from input"
                
        except Exception as e:
            return f"Error normalizing code: {str(e)}\n{traceback.format_exc()}"
    except Exception as e:
        return f"Error normalizing code: {str(e)}\n{traceback.format_exc()}"

# ----------------------------
# AST-path similarity
# ----------------------------
def extract_ast_paths(node, parent_type=""):
    paths = []
    current_type = type(node).__name__
    path = f"{parent_type}->{current_type}" if parent_type else current_type
    paths.append(path)
    for child in ast.iter_child_nodes(node):
        paths.extend(extract_ast_paths(child, current_type))
    return paths

def ast_path_similarity(code1, code2):
    try:
        tree1 = ast.parse(code1)
        tree2 = ast.parse(code2)
        paths1 = extract_ast_paths(tree1)
        paths2 = extract_ast_paths(tree2)
        
        # If paths are identical, return 1.0 immediately
        if paths1 == paths2:
            return 1.0
            
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([" ".join(paths1), " ".join(paths2)])
        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return sim
    except:
        return 0

# ----------------------------
# Surface-level token similarity
# ----------------------------
def surface_similarity(code1, code2):
    try:
        # If code is identical, return 1.0 immediately
        if code1.strip() == code2.strip():
            return 1.0
            
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5))
        tfidf = vectorizer.fit_transform([code1, code2])
        return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    except:
        return 0

# ----------------------------
# Enhanced Similarity Calculation
# ----------------------------
def calculate_comprehensive_similarity(code1, code2, normalized1, normalized2):
    # Check if code is identical first
    if code1.strip() == code2.strip():
        return {
            'ast_similarity': 1.0,
            'surface_similarity': 1.0,
            'cfg_similarity': 1.0,
            'semantic_similarity': 1.0,
            'chunk_similarity': 1.0,
            'combined_similarity': 1.0
        }
    
    # Multiple similarity measures
    ast_sim = ast_path_similarity(normalized1, normalized2)
    surface_sim = surface_similarity(code1, code2)
    cfg_sim = cfg_similarity(extract_control_flow(code1), extract_control_flow(code2))
    semantic_sim = semantic_similarity(code1, code2)
    chunk_sim = chunk_based_similarity(code1, code2)
    
    # If normalized code is identical, boost the scores
    if normalized1 == normalized2:
        ast_sim = max(ast_sim, 0.95)
        surface_sim = max(surface_sim, 0.95)
    
    # Weighted combination
    weights = {
        'ast': 0.3,
        'surface': 0.2,
        'cfg': 0.2,
        'semantic': 0.2,
        'chunk': 0.1
    }
    
    combined = (
        ast_sim * weights['ast'] +
        surface_sim * weights['surface'] +
        cfg_sim * weights['cfg'] +
        semantic_sim * weights['semantic'] +
        chunk_sim * weights['chunk']
    )
    
    return {
        'ast_similarity': ast_sim,
        'surface_similarity': surface_sim,
        'cfg_similarity': cfg_sim,
        'semantic_similarity': semantic_sim,
        'chunk_similarity': chunk_sim,
        'combined_similarity': combined
    }

def generate_detailed_analysis(scores, code1, code2):
    analysis = []
    
    if scores['ast_similarity'] > 0.8:
        analysis.append("• Very similar abstract syntax tree structure")
    elif scores['ast_similarity'] > 0.6:
        analysis.append("• Similar program structure and organization")
    
    if scores['semantic_similarity'] > 0.7:
        analysis.append("• Similar algorithmic patterns and operations")
    
    if scores['cfg_similarity'] > 0.7:
        analysis.append("• Similar control flow and execution paths")
    
    if len(code1.split()) > 50 and len(code2.split()) > 50:
        if scores['surface_similarity'] > 0.6:
            analysis.append("• Significant surface-level code similarity")
    
    if scores['chunk_similarity'] > 0.7:
        analysis.append("• Similar code organization into functions/classes/loops")
    
    return "\n".join(analysis) if analysis else "No significant patterns detected"

# ----------------------------
# Flask route
# ----------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    code1 = code2 = ""
    normalized1 = normalized2 = ""
    similarity_scores = {}
    verdict = ""
    detailed_analysis = ""

    if request.method == "POST":
        code1 = request.form.get("code1", "")
        code2 = request.form.get("code2", "")

        reset_global_maps()
        
        # For identical code, use minimal normalization
        if code1.strip() == code2.strip():
            normalized1 = normalize_code(code1, "minimal")
            normalized2 = normalize_code(code2, "minimal")
        else:
            normalized1 = normalize_code(code1, "auto")
            normalized2 = normalize_code(code2, "auto")

        # Calculate comprehensive similarity
        similarity_scores = calculate_comprehensive_similarity(code1, code2, normalized1, normalized2)
        
        # Convert to percentages
        for key in similarity_scores:
            if key != 'combined_similarity':
                similarity_scores[key] = round(similarity_scores[key] * 100, 2)
        
        similarity_scores['combined_percentage'] = round(similarity_scores['combined_similarity'] * 100, 2)
        
        # Generate detailed analysis
        detailed_analysis = generate_detailed_analysis(similarity_scores, code1, code2)

        # Verdict rules
        final_score = similarity_scores['combined_similarity']
        if final_score >= 0.9:
            verdict = "🚨 HIGH PLAGIARISM RISK: Very similar structure, logic, and implementation"
        elif final_score >= 0.7:
            verdict = "⚠️ MODERATE RISK: Significant similarities in structure and logic"
        elif final_score >= 0.5:
            verdict = "🔍 SUSPICIOUS: Some concerning similarities detected"
        elif final_score >= 0.3:
            verdict = "👀 LOW SIMILARITY: Minor similarities, likely coincidental"
        else:
            verdict = "✅ INDEPENDENT: Code appears to be independently written"

    return render_template(
        "enhanced_index.html",
        code1=code1,
        code2=code2,
        normalized1=normalized1,
        normalized2=normalized2,
        similarity_scores=similarity_scores,
        verdict=verdict,
        detailed_analysis=detailed_analysis
    )

# ----------------------------
if __name__ == '__main__':
    app.run(debug=True)
