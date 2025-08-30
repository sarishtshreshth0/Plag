from flask import Flask, render_template, request
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
# Canonical Order Transformer
# ----------------------------
class CanonicalOrder(ast.NodeTransformer):
    def visit_Module(self, node):
        assigns = [stmt for stmt in node.body if isinstance(stmt, ast.Assign)]
        others = [stmt for stmt in node.body if not isinstance(stmt, ast.Assign)]
        
        # Topological-like sorting for assigns
        ordered = []
        assigned_vars = set()
        while assigns:
            progress = False
            for stmt in assigns[:]:
                used_vars = {n.id for n in ast.walk(stmt.value) if isinstance(n, ast.Name)}
                if used_vars <= assigned_vars:
                    ordered.append(stmt)
                    for target in stmt.targets:
                        if isinstance(target, ast.Name):
                            assigned_vars.add(target.id)
                    assigns.remove(stmt)
                    progress = True
            if not progress:
                ordered.extend(assigns)
                break
        
        node.body = ordered + others
        return node

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
# Normalization Pipeline
# ----------------------------
def normalize_code(code):
    try:
        tree = ast.parse(code)
        tree = CanonicalOrder().visit(tree)
        tree = RenameVariablesFunctionsGlobal().visit(tree)
        tree = LiteralNormalizer().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception as e:
        return f"Error normalizing code: {e}"

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
    tree1 = ast.parse(code1)
    tree2 = ast.parse(code2)
    paths1 = extract_ast_paths(tree1)
    paths2 = extract_ast_paths(tree2)
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([" ".join(paths1), " ".join(paths2)])
    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return sim

# ----------------------------
# Surface-level token similarity
# ----------------------------
def surface_similarity(code1, code2):
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5))
    tfidf = vectorizer.fit_transform([code1, code2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

# ----------------------------
# Flask route
# ----------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    code1 = code2 = ""
    normalized1 = normalized2 = ""
    similarity_score = surface_score = None
    verdict = ""

    if request.method == "POST":
        code1 = request.form.get("code1", "")
        code2 = request.form.get("code2", "")

        reset_global_maps()
        normalized1 = normalize_code(code1)
        normalized2 = normalize_code(code2)

        similarity_score = round(ast_path_similarity(normalized1, normalized2)*100,2)
        surface_score = round(surface_similarity(code1, code2)*100,2)

        # Verdict rules
        if similarity_score >= 95 and surface_score >= 80:
            verdict = "High chance of direct copy-paste"
        elif similarity_score >= 90 and surface_score < 80:
            verdict = "Likely rephrased / rewritten"
        else:
            verdict = "Independent code / low similarity"

    return render_template(
        "index.html",
        code1=code1,
        code2=code2,
        normalized1=normalized1,
        normalized2=normalized2,
        similarity_score=similarity_score,
        surface_score=surface_score,
        verdict=verdict
    )

# ----------------------------
if __name__ == '__main__':
    app.run(debug=True)
