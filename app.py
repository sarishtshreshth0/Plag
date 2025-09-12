from flask import Flask, render_template, request
import ast
import networkx as nx
import traceback
import re
import math
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import textwrap
from sklearn.preprocessing import StandardScaler

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
# ML Model Initialization
# ----------------------------
# We'll create a simple ML model for plagiarism detection
try:
    # Try to load a pre-trained model if available
    ml_model = joblib.load('plagiarism_model.pkl')
    ml_scaler = joblib.load('plagiarism_scaler.pkl')
except:
    # Initialize new model if not available
    ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
    ml_scaler = StandardScaler()
    model_trained = False

# ----------------------------
# Dataset for training (would be populated with real data in production)
# ----------------------------
# In a real scenario, you would have a dataset of code pairs labeled as plagiarized or not
# For demonstration, we'll create a simple synthetic dataset
def create_synthetic_dataset():
    # This is just a placeholder - in a real scenario, you'd use a proper dataset
    features = []
    labels = []
    
    # Generate some synthetic data
    for i in range(100):
        # Simulate feature vectors (5 features)
        features.append([np.random.random() for _ in range(5)])
        labels.append(np.random.randint(0, 2))
    
    return np.array(features), np.array(labels)

# Train the model if not already trained
if not 'model_trained' in locals() or not model_trained:
    try:
        X, y = create_synthetic_dataset()
        X_scaled = ml_scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        ml_model.fit(X_train, y_train)
        model_trained = True
        # Save the model for future use
        joblib.dump(ml_model, 'plagiarism_model.pkl')
        joblib.dump(ml_scaler, 'plagiarism_scaler.pkl')
    except:
        model_trained = False

# ----------------------------
# ML-Based Feature Extraction
# ----------------------------
def extract_ml_features(code1, code2, normalized1, normalized2):
    """Extract features for ML model"""
    features = []
    
    # 1. AST Path Similarity
    ast_sim = ast_path_similarity(normalized1, normalized2)
    features.append(ast_sim)
    
    # 2. Surface Similarity
    surface_sim = surface_similarity(code1, code2)
    features.append(surface_sim)
    
    # 3. CFG Similarity
    cfg_sim = cfg_similarity(extract_control_flow(code1), extract_control_flow(code2))
    features.append(cfg_sim)
    
    # 4. Semantic Similarity
    semantic_sim = semantic_similarity(code1, code2)
    features.append(semantic_sim)
    
    # 5. Chunk Similarity
    chunk_sim = chunk_based_similarity(code1, code2)
    features.append(chunk_sim)
    
    # 6. Token-based TF-IDF Similarity
    tfidf_sim = calculate_tfidf_similarity(code1, code2)
    features.append(tfidf_sim)
    
    # 7. Code Length Ratio
    len1, len2 = len(code1.split()), len(code2.split())
    len_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
    features.append(len_ratio)
    
    # 8. Unique Identifier Ratio
    unique_ratio = calculate_identifier_similarity(code1, code2)
    features.append(unique_ratio)
    
    return np.array(features).reshape(1, -1)

def calculate_tfidf_similarity(code1, code2):
    """Calculate similarity using TF-IDF vectors"""
    try:
        # Tokenize the code
        tokens1 = tokenize_code(code1)
        tokens2 = tokenize_code(code2)
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([' '.join(tokens1), ' '.join(tokens2)])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except:
        return 0

def tokenize_code(code):
    """Tokenize code for TF-IDF analysis"""
    # Remove comments and strings for better tokenization
    code = re.sub(r'#.*', '', code)  # Remove comments
    code = re.sub(r'\"\"\"[\s\S]*?\"\"\"', '', code)  # Remove multi-line strings
    code = re.sub(r'\'\'\'[\s\S]*?\'\'\'', '', code)  # Remove multi-line strings
    code = re.sub(r'\"[^\"]*\"', '', code)  # Remove strings
    code = re.sub(r'\'[^\']*\'', '', code)  # Remove strings
    
    # Tokenize
    tokens = re.findall(r'\b\w+\b', code)
    return tokens

def calculate_identifier_similarity(code1, code2):
    """Calculate similarity based on unique identifiers"""
    try:
        # Extract identifiers from code
        tree1 = ast.parse(code1)
        tree2 = ast.parse(code2)
        
        identifiers1 = extract_identifiers(tree1)
        identifiers2 = extract_identifiers(tree2)
        
        if not identifiers1 or not identifiers2:
            return 0
        
        # Calculate Jaccard similarity
        set1 = set(identifiers1)
        set2 = set(identifiers2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0
    except:
        return 0

def extract_identifiers(tree):
    """Extract all identifiers from AST"""
    identifiers = []
    
    class IdentifierExtractor(ast.NodeVisitor):
        def visit_Name(self, node):
            identifiers.append(node.id)
            self.generic_visit(node)
        
        def visit_FunctionDef(self, node):
            identifiers.append(node.name)
            self.generic_visit(node)
        
        def visit_ClassDef(self, node):
            identifiers.append(node.name)
            self.generic_visit(node)
        
        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name):
                identifiers.append(node.value.id + '.' + node.attr)
            self.generic_visit(node)
    
    extractor = IdentifierExtractor()
    extractor.visit(tree)
    return identifiers

# ----------------------------
# ML-Based Prediction
# ----------------------------
def ml_predict_plagiarism(features):
    """Use ML model to predict plagiarism probability"""
    if not model_trained:
        # Fallback to traditional method if model is not trained
        return np.mean(features[0][:5])  # Average of first 5 features
    
    try:
        # Scale features
        features_scaled = ml_scaler.transform(features)
        # Predict probability
        probability = ml_model.predict_proba(features_scaled)[0][1]
        return probability
    except:
        return np.mean(features[0][:5])  # Fallback

# ----------------------------
# Whitespace Normalization
# ----------------------------
def normalize_whitespace(code):
    """Remove extra whitespace and standardize formatting"""
    if not code:
        return code
    
    # Remove trailing whitespace from each line
    lines = [line.rstrip() for line in code.split('\n')]
    
    # Remove empty lines and lines with only whitespace
    lines = [line for line in lines if line.strip()]
    
    # Standardize spaces around operators and punctuation
    normalized_lines = []
    
    for line in lines:
        # Spaces around assignment operators
        line = re.sub(r'\s*=\s*', ' = ', line)
        
        # Spaces around arithmetic operators
        line = re.sub(r'\s*\+\s*', ' + ', line)
        line = re.sub(r'\s*\-\s*', ' - ', line)
        line = re.sub(r'\s*\*\s*', ' * ', line)
        line = re.sub(r'\s*/\s*', ' / ', line)
        line = re.sub(r'\s*%\s*', ' % ', line)
        line = re.sub(r'\s*//\s*', ' // ', line)
        line = re.sub(r'\s*\*\*\s*', ' ** ', line)
        
        # Spaces around comparison operators
        line = re.sub(r'\s*==\s*', ' == ', line)
        line = re.sub(r'\s*!=\s*', ' != ', line)
        line = re.sub(r'\s*<\s*', ' < ', line)
        line = re.sub(r'\s*<=\s*', ' <= ', line)
        line = re.sub(r'\s*>\s*', ' > ', line)
        line = re.sub(r'\s*>=\s*', ' >= ', line)
        
        # Spaces around logical operators
        line = re.sub(r'\s*and\s*', ' and ', line)
        line = re.sub(r'\s*or\s*', ' or ', line)
        line = re.sub(r'\s*not\s*', ' not ', line)
        
        # Spaces around punctuation
        line = re.sub(r'\s*,\s*', ', ', line)
        line = re.sub(r'\s*:\s*', ': ', line)
        line = re.sub(r'\s*;\s*', '; ', line)
        
        # Remove multiple spaces
        line = re.sub(r'\s+', ' ', line)
        
        # Remove spaces at beginning and end of line
        line = line.strip()
        
        normalized_lines.append(line)
    
    return '\n'.join(normalized_lines)

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
# Structure Normalization
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
# Text Similarity Functions
# ----------------------------
def text_similarity(text1, text2):
    """Calculate text similarity using cosine similarity of word vectors"""
    if not text1 or not text2:
        return 0
    
    # Tokenize texts
    def get_tokens(text):
        return re.findall(r'\w+', text.lower())
    
    tokens1 = get_tokens(text1)
    tokens2 = get_tokens(text2)
    
    if not tokens1 or not tokens2:
        return 0
    
    # Create vocabulary
    vocabulary = set(tokens1 + tokens2)
    
    # Create word frequency vectors
    vec1 = Counter(tokens1)
    vec2 = Counter(tokens2)
    
    # Calculate cosine similarity
    dot_product = sum(vec1.get(word, 0) * vec2.get(word, 0) for word in vocabulary)
    magnitude1 = math.sqrt(sum(vec1.get(word, 0)**2 for word in vocabulary))
    magnitude2 = math.sqrt(sum(vec2.get(word, 0)**2 for word in vocabulary))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    
    return dot_product / (magnitude1 * magnitude2)

def ngram_similarity(text1, text2, n=3):
    """Calculate similarity using n-grams"""
    if not text1 or not text2:
        return 0
    
    def get_ngrams(text, n):
        return [text[i:i+n] for i in range(len(text)-n+1)]
    
    ngrams1 = get_ngrams(text1, n)
    ngrams2 = get_ngrams(text2, n)
    
    if not ngrams1 or not ngrams2:
        return 0
    
    set1 = set(ngrams1)
    set2 = set(ngrams2)
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0

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
            
        # Use text similarity instead of TF-IDF
        text1 = " ".join(paths1)
        text2 = " ".join(paths2)
        return text_similarity(text1, text2)
    except:
        return 0

# ----------------------------
# Surface-level token similarity
# ----------------------------
def surface_similarity(code1, code2):
    try:
        # First normalize whitespace
        code1_clean = normalize_whitespace(code1)
        code2_clean = normalize_whitespace(code2)
        
        # If cleaned code is identical, return 1.0 immediately
        if code1_clean.strip() == code2_clean.strip():
            return 1.0
            
        # Use combination of text similarity and ngram similarity
        text_sim = text_similarity(code1_clean, code2_clean)
        ngram_sim = ngram_similarity(code1_clean, code2_clean, 3)
        
        return (text_sim + ngram_sim) / 2
    except:
        return 0

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
    
    # Manual cosine similarity calculation
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(v * v for v in vec1))
    magnitude2 = math.sqrt(sum(v * v for v in vec2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    
    return dot_product / (magnitude1 * magnitude2)

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
# Normalization Pipeline
# ----------------------------
def normalize_code(code, normalization_level="auto"):
    try:
        # FIRST: Normalize whitespace
        code = textwrap.dedent(code).strip()
        
        # THEN: Parse and apply other normalizations
        tree = ast.parse(code)
        
        # Auto-detect normalization level based on code size
        if normalization_level == "auto":
            code_size = len(code.split('\n'))
            if code_size <= 10:  # Small code
                normalization_level = "medium"
            elif code_size <= 50:  # Medium code
                normalization_level = "medium"
            else:  # Large code
                normalization_level = "full"
        
        # Apply appropriate normalization
        if normalization_level == "full":
            tree = RenameVariablesFunctionsGlobal().visit(tree)
            tree = LiteralNormalizer().visit(tree)
            tree = StructureNormalizer().visit(tree)
        elif normalization_level == "medium":
            tree = RenameVariablesFunctionsGlobal().visit(tree)
            tree = LiteralNormalizer().visit(tree)
        else:  # minimal (but still include variable renaming)
            tree = RenameVariablesFunctionsGlobal().visit(tree)
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
                    stmt_code = '\n'.join(current_statement)
                    stmt_tree = ast.parse(stmt_code)
                    parsed_statements.append(stmt_tree)
                    current_statement = []
                except SyntaxError:
                    continue
            
            if parsed_statements:
                new_module = ast.Module(body=[], type_ignores=[])
                for stmt in parsed_statements:
                    if isinstance(stmt, ast.Module):
                        new_module.body.extend(stmt.body)
                
                tree = RenameVariablesFunctionsGlobal().visit(new_module)
                tree = LiteralNormalizer().visit(tree)
                ast.fix_missing_locations(tree)
                return ast.unparse(tree)
            else:
                return f"Error: Could not parse any valid Python code from input"
                
        except Exception as e:
            return f"Error normalizing code: {str(e)}\n{traceback.format_exc()}"
    except Exception as e:
        return f"Error normalizing code: {str(e)}\n{traceback.format_exc()}"
# ----------------------------
# Enhanced Similarity Calculation with ML
# ----------------------------
def calculate_comprehensive_similarity(code1, code2, normalized1, normalized2):
    # Check if code is identical first (after whitespace normalization)
    code1_clean = normalize_whitespace(code1)
    code2_clean = normalize_whitespace(code2)
    
    if code1_clean.strip() == code2_clean.strip():
        return {
            'ast_similarity': 1.0,
            'surface_similarity': 1.0,
            'cfg_similarity': 1.0,
            'semantic_similarity': 1.0,
            'chunk_similarity': 1.0,
            'tfidf_similarity': 1.0,
            'identifier_similarity': 1.0,
            'length_ratio': 1.0,
            'combined_similarity': 1.0,
            'ml_plagiarism_probability': 1.0
        }
    
    # Extract ML features
    ml_features = extract_ml_features(code1, code2, normalized1, normalized2)
    
    # Multiple similarity measures
    ast_sim = ast_path_similarity(normalized1, normalized2)
    surface_sim = surface_similarity(code1, code2)
    cfg_sim = cfg_similarity(extract_control_flow(code1), extract_control_flow(code2))
    semantic_sim = semantic_similarity(code1, code2)
    chunk_sim = chunk_based_similarity(code1, code2)
    tfidf_sim = calculate_tfidf_similarity(code1, code2)
    
    # Length ratio
    len1, len2 = len(code1.split()), len(code2.split())
    len_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
    
    # Identifier similarity
    identifier_sim = calculate_identifier_similarity(code1, code2)
    
    # If normalized code is identical, boost the scores
    if normalized1 == normalized2:
        ast_sim = max(ast_sim, 0.95)
        surface_sim = max(surface_sim, 0.95)
    
    # ML-based plagiarism probability
    ml_probability = ml_predict_plagiarism(ml_features)
    
    # Weighted combination
    weights = {
        'ast': 0.2,
        'surface': 0.15,
        'cfg': 0.15,
        'semantic': 0.15,
        'chunk': 0.1,
        'tfidf': 0.1,
        'identifier': 0.1,
        'length': 0.05
    }
    
    combined = (
        ast_sim * weights['ast'] +
        surface_sim * weights['surface'] +
        cfg_sim * weights['cfg'] +
        semantic_sim * weights['semantic'] +
        chunk_sim * weights['chunk'] +
        tfidf_sim * weights['tfidf'] +
        identifier_sim * weights['identifier'] +
        len_ratio * weights['length']
    )
    
    # Adjust based on ML prediction
    combined = 0.7 * combined + 0.3 * ml_probability
    
    return {
        'ast_similarity': ast_sim,
        'surface_similarity': surface_sim,
        'cfg_similarity': cfg_sim,
        'semantic_similarity': semantic_sim,
        'chunk_similarity': chunk_sim,
        'tfidf_similarity': tfidf_sim,
        'identifier_similarity': identifier_sim,
        'length_ratio': len_ratio,
        'combined_similarity': combined,
        'ml_plagiarism_probability': ml_probability
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
    
    if scores['ml_plagiarism_probability'] > 0.7:
        analysis.append("• Machine learning model indicates high probability of plagiarism")
    
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
    ml_features = []

    if request.method == "POST":
        code1 = request.form.get("code1", "")
        code2 = request.form.get("code2", "")

        # Normalize Code 1
        reset_global_maps()
        normalized1 = normalize_code(code1, "medium")

        # Normalize Code 2 (independent reset)
        reset_global_maps()
        normalized2 = normalize_code(code2, "medium")

        # Calculate comprehensive similarity
        similarity_scores = calculate_comprehensive_similarity(code1, code2, normalized1, normalized2)
        
        # Extract ML features for display
        ml_features = extract_ml_features(code1, code2, normalized1, normalized2)[0].tolist()
        feature_names = [
            'AST Similarity', 'Surface Similarity', 'CFG Similarity', 
            'Semantic Similarity', 'Chunk Similarity', 'TF-IDF Similarity',
            'Length Ratio', 'Identifier Similarity'
        ]
        ml_features_display = list(zip(feature_names, ml_features))
        
        # Convert to percentages
        for key in similarity_scores:
            if key != 'combined_similarity' and key != 'ml_plagiarism_probability':
                similarity_scores[key] = round(similarity_scores[key] * 100, 2)
        
        similarity_scores['combined_percentage'] = round(similarity_scores['combined_similarity'] * 100, 2)
        similarity_scores['ml_plagiarism_percentage'] = round(similarity_scores['ml_plagiarism_probability'] * 100, 2)
        
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
        "index.html",
        code1=code1,
        code2=code2,
        normalized1=normalized1,
        normalized2=normalized2,
        similarity_scores=similarity_scores,
        verdict=verdict,
        detailed_analysis=detailed_analysis,
        ml_features=ml_features_display if 'ml_features_display' in locals() else []
    )

# ----------------------------
if __name__ == '__main__':
    app.run(debug=True)
