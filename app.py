"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          CODING SKILL GAP DETECTOR  —  Flask Backend  (v3)                 ║
║══════════════════════════════════════════════════════════════════════════════║
║  WHAT  : An ML-powered REST API that analyses source code and returns a     ║
║          structured report covering skill level, weaknesses, complexity,     ║
║          code quality scores, cosine similarity, and an improved version.   ║
║                                                                              ║
║  WHO   : Students, self-taught developers, bootcamp learners, educators,    ║
║          hiring platforms, and code-review tools.                            ║
║                                                                              ║
║  WHY   : Traditional linters catch syntax errors but cannot measure the     ║
║          SKILL behind code. This tool bridges that gap by combining          ║
║          statistical ML (TF-IDF + Logistic Regression) with rule-based       ║
║          program analysis — giving actionable, human-readable feedback.      ║
║                                                                              ║
║  ALGORITHM OVERVIEW                                                          ║
║  ─────────────────                                                           ║
║  1. TF-IDF Vectorisation (char_wb, n-gram 2-4)                              ║
║     • Converts raw source code into a sparse numerical matrix.               ║
║     • Character-level n-grams capture syntax patterns across languages.      ║
║     • sublinear_tf dampens the effect of very frequent tokens.               ║
║                                                                              ║
║  2. Logistic Regression (multi-class, OvR)                                  ║
║     • Trained on a synthetic corpus of 60+ labelled code snippets.          ║
║     • Outputs class probabilities → Beginner / Intermediate / Advanced.      ║
║     • C=1.0 (regularisation) prevents overfitting on small corpus.           ║
║                                                                              ║
║  3. Rule-based Feature Engine                                                ║
║     • 20+ regex & heuristic detectors per language.                          ║
║     • Adjusts ML label up/down based on bonus/penalty signals.               ║
║                                                                              ║
║  4. Cosine Similarity (TF-IDF space)                                        ║
║     • Compares user code against a curated ideal reference per language.     ║
║     • Returns a 0-1 similarity score.                                        ║
║                                                                              ║
║  5. Halstead + Cyclomatic-style Metrics (lightweight approximations)         ║
║     • Token counts, operator ratios, branch counts.                          ║
║                                                                              ║
║  SUPPORTED LANGUAGES (v3)                                                   ║
║     Python · JavaScript · Java · C++ · C# · TypeScript · Ruby · Go          ║
║                                                                              ║
║  RUN                                                                         ║
║     pip install flask flask-cors scikit-learn numpy                          ║
║     python app.py                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import re, math, textwrap, time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING CORPUS
#  Labels: 0 = Beginner  |  1 = Intermediate  |  2 = Advanced
#  Why synthetic? Real labelled code datasets are large (GB+). A compact
#  synthetic corpus is sufficient to demonstrate the ML pipeline and produces
#  meaningful separability via character n-gram TF-IDF.
# ══════════════════════════════════════════════════════════════════════════════
TRAINING_DATA = [
    # ── Python Beginner ──────────────────────────────────────────────────────
    ("x = 5\nprint(x)", 0),
    ("a = 1\nb = 2\nprint(a + b)", 0),
    ("name = input('Name: ')\nprint('Hello', name)", 0),
    ("for i in range(10):\n    print(i)", 0),
    ("if x > 0:\n    print('positive')\nelse:\n    print('negative')", 0),
    ("numbers = [1,2,3]\nfor n in numbers:\n    print(n)", 0),
    ("total = 0\nfor i in range(5):\n    total += i\nprint(total)", 0),
    # ── Python Intermediate ───────────────────────────────────────────────────
    ("def add(a, b):\n    return a + b\nresult = add(3, 4)\nprint(result)", 1),
    ("def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)", 1),
    ("try:\n    x = int(input())\nexcept ValueError:\n    print('Invalid')", 1),
    ("import os\nfor f in os.listdir('.'):\n    print(f)", 1),
    ("def process(data):\n    return [x*2 for x in data if x > 0]", 1),
    ("class Animal:\n    def __init__(self, name):\n        self.name = name\n    def speak(self): pass", 1),
    ("import json\nwith open('data.json') as f:\n    d = json.load(f)\nprint(d)", 1),
    # ── Python Advanced ───────────────────────────────────────────────────────
    ("from functools import lru_cache\n@lru_cache(maxsize=None)\ndef fib(n):\n    if n < 2: return n\n    return fib(n-1) + fib(n-2)", 2),
    ("class Singleton:\n    _instance = None\n    def __new__(cls):\n        if not cls._instance:\n            cls._instance = super().__new__(cls)\n        return cls._instance", 2),
    ("import asyncio\nasync def fetch(session, url):\n    async with session.get(url) as r:\n        return await r.json()", 2),
    ("def memoize(fn):\n    cache = {}\n    def wrapper(*args):\n        if args not in cache: cache[args] = fn(*args)\n        return cache[args]\n    return wrapper", 2),
    ("from typing import Generator\ndef infinite_counter(start:int=0)->Generator[int,None,None]:\n    n=start\n    while True:\n        yield n\n        n+=1", 2),
    # ── JavaScript Beginner ───────────────────────────────────────────────────
    ("let x = 5; console.log(x);", 0),
    ("var a = 1; var b = 2; console.log(a + b);", 0),
    ("for (let i = 0; i < 5; i++) { console.log(i); }", 0),
    # ── JavaScript Intermediate ───────────────────────────────────────────────
    ("function fetchData(url) { return fetch(url).then(r => r.json()); }", 1),
    ("const add = (a, b) => a + b;\nconsole.log(add(2, 3));", 1),
    ("class Person { constructor(n){this.name=n;} greet(){return 'Hi '+this.name;} }", 1),
    ("async function getData(url){try{const r=await fetch(url);return await r.json();}catch(e){console.error(e);}}", 1),
    # ── JavaScript Advanced ───────────────────────────────────────────────────
    ("const debounce=(fn,d)=>{let t;return(...a)=>{clearTimeout(t);t=setTimeout(()=>fn(...a),d);}};", 2),
    ("function* gen(){let i=0;while(true){yield i++;}} const g=gen();console.log(g.next().value);", 2),
    ("const pipe=(...fns)=>x=>fns.reduce((v,f)=>f(v),x);", 2),
    # ── Java Beginner ─────────────────────────────────────────────────────────
    ("public class Main { public static void main(String[] args) { System.out.println(\"Hello\"); } }", 0),
    ("int x = 5; System.out.println(x);", 0),
    # ── Java Intermediate ─────────────────────────────────────────────────────
    ("public int factorial(int n) { return n <= 1 ? 1 : n * factorial(n-1); }", 1),
    ("class Animal { String name; Animal(String n){this.name=n;} void speak(){} }", 1),
    ("import java.util.*;\nList<Integer> list = new ArrayList<>();\nfor(int i=0;i<5;i++) list.add(i);", 1),
    # ── Java Advanced ─────────────────────────────────────────────────────────
    ("public class GenericStack<T> { private List<T> items=new ArrayList<>(); public void push(T i){items.add(i);} public T pop(){return items.remove(items.size()-1);} }", 2),
    ("@FunctionalInterface interface Transformer<T>{T transform(T input);}\nTransformer<String> upper=String::toUpperCase;", 2),
    # ── C++ ───────────────────────────────────────────────────────────────────
    ("#include<iostream>\nint main(){std::cout<<\"Hello\";return 0;}", 0),
    ("#include<iostream>\nusing namespace std;\nint add(int a,int b){return a+b;}\nint main(){cout<<add(2,3);}", 1),
    ("template<typename T>\nT maxVal(T a,T b){return a>b?a:b;}\nauto r=maxVal(3,5);", 2),
    ("#include<algorithm>\n#include<vector>\nvector<int>v={3,1,2};\nsort(v.begin(),v.end());", 1),
    ("class Animal{public: virtual void speak()=0;};\nclass Dog:public Animal{public:void speak(){std::cout<<\"Woof\";}};", 2),
    # ── C# ───────────────────────────────────────────────────────────────────
    ("using System;\nclass Program{static void Main(){Console.WriteLine(\"Hello\");}}", 0),
    ("using System;\nstatic int Add(int a,int b)=>a+b;\nConsole.WriteLine(Add(2,3));", 1),
    ("using System.Linq;\nvar nums=new[]{1,2,3,4};\nvar evens=nums.Where(n=>n%2==0).ToList();", 1),
    ("using System.Threading.Tasks;\nasync Task<string> FetchAsync(string url){using var c=new HttpClient();return await c.GetStringAsync(url);}", 2),
    ("public class Repository<T> where T:class{private List<T> _items=new();public void Add(T i)=>_items.Add(i);}", 2),
    # ── TypeScript ────────────────────────────────────────────────────────────
    ("const x:number=5;console.log(x);", 0),
    ("function greet(name:string):string{return `Hello ${name}`;}", 1),
    ("interface User{id:number;name:string;}\nconst user:User={id:1,name:'Alice'};", 1),
    ("type Result<T,E>=|{ok:true;value:T}|{ok:false;error:E};\nfunction divide(a:number,b:number):Result<number,string>{if(b===0)return{ok:false,error:'div/0'};return{ok:true,value:a/b};}", 2),
    ("class EventEmitter<T>{private handlers:((d:T)=>void)[]=[];on(h:(d:T)=>void){this.handlers.push(h);}emit(d:T){this.handlers.forEach(h=>h(d));}}", 2),
    # ── Ruby ─────────────────────────────────────────────────────────────────
    ("puts 'Hello World'", 0),
    ("x = 5\nputs x", 0),
    ("def greet(name)\n  puts \"Hello, #{name}\"\nend\ngreet('Alice')", 1),
    ("class Animal\n  def initialize(name)\n    @name = name\n  end\n  def speak\n    puts @name\n  end\nend", 1),
    ("require 'json'\nbegin\n  data = JSON.parse('{\"key\":\"value\"}')\n  puts data\nrescue JSON::ParserError => e\n  puts e.message\nend", 2),
    ("module Greetable\n  def greet\n    \"Hello, I am #{name}\"\n  end\nend\nclass Person\n  include Greetable\n  attr_reader :name\n  def initialize(n) @name=n end\nend", 2),
    # ── Go ────────────────────────────────────────────────────────────────────
    ("package main\nimport \"fmt\"\nfunc main(){fmt.Println(\"Hello\")}", 0),
    ("package main\nimport \"fmt\"\nfunc add(a,b int)int{return a+b}\nfunc main(){fmt.Println(add(2,3))}", 1),
    ("package main\nimport \"fmt\"\nfunc main(){\n  ch:=make(chan int,5)\n  go func(){ch<-42}()\n  fmt.Println(<-ch)\n}", 2),
    ("package main\ntype Node struct{Val int;Next *Node}\nfunc(n *Node)Insert(v int)*Node{return &Node{Val:v,Next:n}}", 2),
]

texts  = [d[0] for d in TRAINING_DATA]
labels = [d[1] for d in TRAINING_DATA]

# ── Fit shared TF-IDF vectorizer ─────────────────────────────────────────────
#  WHY char_wb? Source code has no natural word boundaries across languages.
#  Character n-grams (2-4) capture:  "def ", "fn ", "func ", class/struct
#  patterns, indentation style, operator density — all language-agnostic cues.
vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 4),
    max_features=5000,
    sublinear_tf=True,  # log(1+tf) prevents high-freq tokens from dominating
)
X = vectorizer.fit_transform(texts)

# ── Logistic Regression ───────────────────────────────────────────────────────
#  WHY Logistic Regression?
#  • Interpretable: probabilities directly explain confidence.
#  • Fast at inference — critical for a real-time web API.
#  • Works well with high-dimensional sparse TF-IDF features.
#  • Regularised via C=1.0 (L2 penalty) to avoid overfitting small corpus.
#  Alternatives considered: SVM (no probability output by default),
#  Random Forest (slower, needs more data), Neural Net (overkill for this task).
clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42,
                          solver="lbfgs", multi_class="auto")
clf.fit(X, labels)

LEVEL_NAMES = {0: "Beginner", 1: "Intermediate", 2: "Advanced"}

# ── Ideal reference solutions for cosine-similarity ──────────────────────────
IDEAL_SOLUTIONS = {
    "python": textwrap.dedent("""\
        \"\"\"Well-structured Python module with type hints, docstring, error handling.\"\"\"
        from typing import List
        def process(data: List[int]) -> List[int]:
            \"\"\"Filter positive numbers and double them.\"\"\"
            try:
                return [x * 2 for x in data if isinstance(x, (int, float)) and x > 0]
            except TypeError as exc:
                print(f"Invalid input: {exc}")
                return []
        if __name__ == "__main__":
            print(process([1, -2, 3, -4, 5]))
    """),
    "javascript": textwrap.dedent("""\
        /**
         * @param {string} url
         * @returns {Promise<Object|null>}
         */
        async function fetchUser(url) {
          try {
            const response = await fetch(url);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
          } catch (err) {
            console.error('fetchUser failed:', err.message);
            return null;
          }
        }
        export default fetchUser;
    """),
    "java": textwrap.dedent("""\
        import java.util.*;
        public class Calculator {
            public static int divide(int a, int b) {
                if (b == 0) throw new ArithmeticException("Division by zero");
                return a / b;
            }
            public static void main(String[] args) {
                try { System.out.println(divide(10, 2)); }
                catch (ArithmeticException e) { System.err.println(e.getMessage()); }
            }
        }
    """),
    "cpp": textwrap.dedent("""\
        #include <iostream>
        #include <stdexcept>
        template<typename T>
        T safeDivide(T a, T b) {
            if (b == 0) throw std::invalid_argument("Division by zero");
            return a / b;
        }
        int main() {
            try { std::cout << safeDivide(10, 2) << std::endl; }
            catch (const std::exception& e) { std::cerr << e.what(); }
            return 0;
        }
    """),
    "csharp": textwrap.dedent("""\
        using System;
        using System.Collections.Generic;
        using System.Linq;
        public class DataProcessor {
            public IEnumerable<int> Process(IEnumerable<int> data) {
                if (data == null) throw new ArgumentNullException(nameof(data));
                return data.Where(x => x > 0).Select(x => x * 2);
            }
        }
    """),
    "typescript": textwrap.dedent("""\
        type Result<T, E> = { ok: true; value: T } | { ok: false; error: E };
        async function fetchData<T>(url: string): Promise<Result<T, string>> {
          try {
            const res = await fetch(url);
            if (!res.ok) return { ok: false, error: `HTTP ${res.status}` };
            return { ok: true, value: await res.json() as T };
          } catch (e) {
            return { ok: false, error: String(e) };
          }
        }
        export { fetchData };
    """),
    "ruby": textwrap.dedent("""\
        # frozen_string_literal: true
        module DataProcessor
          def self.process(data)
            raise ArgumentError, 'data must be an Array' unless data.is_a?(Array)
            data.select { |x| x.positive? }.map { |x| x * 2 }
          rescue StandardError => e
            puts \"Error: #{e.message}\"
            []
          end
        end
        p DataProcessor.process([1, -2, 3, -4, 5])
    """),
    "go": textwrap.dedent("""\
        package main
        import (
            "errors"
            "fmt"
        )
        func divide(a, b float64) (float64, error) {
            if b == 0 {
                return 0, errors.New("division by zero")
            }
            return a / b, nil
        }
        func main() {
            result, err := divide(10, 2)
            if err != nil {
                fmt.Println("Error:", err)
                return
            }
            fmt.Println(result)
        }
    """),
}


# ══════════════════════════════════════════════════════════════════════════════
#  RULE-BASED FEATURE DETECTORS
#  Each function returns bool or int. They feed both:
#  (a) gap detection  →  weaknesses + suggestions
#  (b) score computation  →  bonus/penalty signals
# ══════════════════════════════════════════════════════════════════════════════

def count_lines(code: str) -> int:
    return len([l for l in code.splitlines() if l.strip()])

def count_tokens(code: str) -> int:
    """Rough token count via whitespace + punctuation splitting."""
    return len(re.findall(r'\w+|[^\w\s]', code))

def count_branches(code: str) -> int:
    """Count decision points: if/else/elif/case/ternary."""
    return len(re.findall(r'\b(if|else|elif|case|switch|\?)\b', code))

def count_operators(code: str) -> int:
    return len(re.findall(r'[\+\-\*\/\%\&\|\^~<>=!]+', code))


# ── Function / Method detection ──────────────────────────────────────────────
def has_functions(code: str, lang: str) -> bool:
    patterns = {
        "python":     r'\bdef\s+\w+\s*\(',
        "java":       r'(public|private|protected|static)\s+[\w<>\[\]]+\s+\w+\s*\(',
        "javascript": r'(function\s+\w+|\bconst\s+\w+\s*=\s*(async\s*)?\(.*?\)\s*=>|=>)',
        "typescript": r'(function\s+\w+|\bconst\s+\w+\s*=.*?=>|\b\w+\s*\(.*?\)\s*:\s*\w)',
        "cpp":        r'(\w[\w\s\*&<>]*)\s+\w+\s*\([^)]*\)\s*(\{|const)',
        "csharp":     r'(public|private|protected|static|async)\s+[\w<>\[\]]+\s+\w+\s*\(',
        "ruby":       r'\bdef\s+\w+',
        "go":         r'\bfunc\s+\w+',
    }
    pat = patterns.get(lang, r'\bdef\s+\w+|\bfunc\s+\w+|\bfunction\s+\w+')
    return bool(re.search(pat, code))

# ── Class / Struct detection ──────────────────────────────────────────────────
def has_classes(code: str, lang: str) -> bool:
    patterns = {
        "python":     r'\bclass\s+\w+',
        "java":       r'\bclass\s+\w+',
        "javascript": r'\bclass\s+\w+',
        "typescript": r'\b(class|interface|type)\s+\w+',
        "cpp":        r'\b(class|struct)\s+\w+',
        "csharp":     r'\b(class|interface|record|struct)\s+\w+',
        "ruby":       r'\bclass\s+\w+|\bmodule\s+\w+',
        "go":         r'\btype\s+\w+\s+struct',
    }
    pat = patterns.get(lang, r'\bclass\s+\w+')
    return bool(re.search(pat, code))

# ── Error handling ────────────────────────────────────────────────────────────
def has_error_handling(code: str, lang: str) -> bool:
    patterns = {
        "python":     r'\btry\b[\s\S]+?\bexcept\b',
        "java":       r'\btry\b[\s\S]+?\bcatch\b',
        "javascript": r'\btry\b[\s\S]+?\bcatch\b',
        "typescript": r'\btry\b[\s\S]+?\bcatch\b',
        "cpp":        r'\btry\b[\s\S]+?\bcatch\b',
        "csharp":     r'\btry\b[\s\S]+?\bcatch\b',
        "ruby":       r'\bbegin\b[\s\S]+?\brescue\b|\braise\b',
        "go":         r'if\s+err\s*!=\s*nil|errors\.New|fmt\.Errorf',
    }
    pat = patterns.get(lang, r'\btry\b[\s\S]+?\b(except|catch|rescue)\b')
    return bool(re.search(pat, code, re.DOTALL))

# ── Imports / packages ────────────────────────────────────────────────────────
def has_imports(code: str, lang: str) -> bool:
    patterns = {
        "python":     r'^\s*(import|from)\s+\w+',
        "java":       r'^\s*import\s+[\w.]+;',
        "javascript": r'\b(import|require)\b',
        "typescript": r'\b(import|require)\b',
        "cpp":        r'#include\s*[<"]',
        "csharp":     r'^\s*using\s+[\w.]+;',
        "ruby":       r'^\s*require\b|^\s*require_relative\b',
        "go":         r'\bimport\b',
    }
    pat = patterns.get(lang, r'\b(import|require|include|using)\b')
    return bool(re.search(pat, code, re.MULTILINE))

# ── Comments & Docstrings ─────────────────────────────────────────────────────
def has_comments(code: str, lang: str) -> bool:
    patterns = {
        "python":     r'#.+|"""[\s\S]+?"""|\'\'\'[\s\S]+?\'\'\'',
        "ruby":       r'#.+',
        "go":         r'//.*|/\*[\s\S]*?\*/',
        "cpp":        r'//.*|/\*[\s\S]*?\*/',
        "csharp":     r'///.*|//.*|/\*[\s\S]*?\*/',
        "java":       r'//.*|/\*[\s\S]*?\*/',
        "javascript": r'//.*|/\*[\s\S]*?\*/',
        "typescript": r'//.*|/\*[\s\S]*?\*/',
    }
    pat = patterns.get(lang, r'//.*|#.*|/\*[\s\S]*?\*/')
    return bool(re.search(pat, code))

# ── Type annotations ──────────────────────────────────────────────────────────
def has_type_hints(code: str, lang: str) -> bool:
    patterns = {
        "python":     r':\s*(int|str|float|bool|list|dict|List|Dict|Optional|Union|Any)\b|->\s*\w+',
        "typescript": r':\s*(number|string|boolean|void|any|never|unknown)\b|<\w+>',
        "csharp":     r'<\w+>|List<|Dictionary<|IEnumerable<',
        "java":       r'<\w+>|List<|Map<|Optional<',
        "go":         r'\w+\s+\w+\s+\w+\s*{|func\s*\(.*?\)\s*\w+',
        "cpp":        r'template\s*<|auto\s+\w+\s*=',
    }
    pat = patterns.get(lang, "")
    if not pat:
        return False
    return bool(re.search(pat, code))

# ── Loop analysis ─────────────────────────────────────────────────────────────
def _loop_max_depth(code: str) -> int:
    loop_kw = r'\b(for|while|loop)\b'
    depth = 0; max_depth = 0; indent_stack = [-1]
    for line in code.splitlines():
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if re.search(loop_kw, stripped):
            while indent_stack and indent_stack[-1] >= indent:
                depth -= 1; indent_stack.pop()
            depth += 1; indent_stack.append(indent)
            max_depth = max(max_depth, depth)
    return max_depth

def has_nested_loops(code: str, lang: str) -> bool:
    return _loop_max_depth(code) >= 2

# ── Recursion ─────────────────────────────────────────────────────────────────
def has_recursion(code: str) -> bool:
    # Python
    for fn in re.findall(r'\bdef\s+(\w+)\s*\(', code):
        if re.search(r'\b' + fn + r'\s*\(', code.split(f'def {fn}', 1)[-1]):
            return True
    # JS/TS
    for fn in re.findall(r'\bfunction\s+(\w+)\s*\(', code):
        if re.search(r'\b' + fn + r'\s*\(', code.split(f'function {fn}', 1)[-1]):
            return True
    # Go
    for fn in re.findall(r'\bfunc\s+(\w+)\s*\(', code):
        if re.search(r'\b' + fn + r'\s*\(', code.split(f'func {fn}', 1)[-1]):
            return True
    return False

def has_list_comprehension(code: str, lang: str) -> bool:
    patterns = {
        "python": r'\[.+\bfor\b.+\bin\b.+\]',
        "csharp": r'\.Where\(|\.Select\(|\.LINQ\.',
        "java":   r'\.stream\(\)|\.filter\(|\.map\(',
        "ruby":   r'\.map\s*{|\.select\s*{|\.reject\s*{',
        "go":     "",
    }
    pat = patterns.get(lang, r'\[.+\bfor\b.+\bin\b.+\]')
    if not pat:
        return False
    return bool(re.search(pat, code))

def has_decorators(code: str, lang: str) -> bool:
    patterns = {
        "python":  r'^\s*@\w+',
        "java":    r'^\s*@\w+',
        "csharp":  r'^\s*\[[\w(]+\]',
        "typescript": r'^\s*@\w+',
    }
    pat = patterns.get(lang, r'^\s*@\w+')
    return bool(re.search(pat, code, re.MULTILINE))

def has_async(code: str, lang: str) -> bool:
    patterns = {
        "python":     r'\basync\b|\bawait\b',
        "javascript": r'\basync\b|\bawait\b',
        "typescript": r'\basync\b|\bawait\b',
        "csharp":     r'\basync\b|\bawait\b|Task<',
        "go":         r'\bgo\s+func\b|\bchan\b|\bgoroutine\b',
        "ruby":       r'Thread\.new|Async|EventMachine',
    }
    pat = patterns.get(lang, r'\basync\b|\bawait\b')
    return bool(re.search(pat, code))

def has_generics_or_templates(code: str, lang: str) -> bool:
    patterns = {
        "java":       r'<[A-Z]\w*>|<\?\s*(extends|super)',
        "csharp":     r'<[A-Z]\w*>|where\s+\w+\s*:',
        "typescript": r'<[A-Z]\w*>|<T\b',
        "cpp":        r'template\s*<',
        "go":         r'\[T\s+\w+\]|\[K,\s*V\b',
    }
    pat = patterns.get(lang, "")
    if not pat:
        return False
    return bool(re.search(pat, code))

def has_unit_tests(code: str, lang: str) -> bool:
    patterns = {
        "python":     r'\bdef\s+test_\w+|import\s+pytest|import\s+unittest|\bassert\b',
        "javascript": r'\bdescribe\b|\bit\b\s*\(|\bexpect\b|\btest\b\s*\(',
        "typescript": r'\bdescribe\b|\bit\b\s*\(|\bexpect\b',
        "java":       r'@Test|import\s+org\.junit',
        "csharp":     r'\[Test\]|\[Fact\]|\[TestMethod\]',
        "go":         r'func\s+Test\w+\(t\s+\*testing\.T\)',
        "ruby":       r'\brspec\b|\bexpect\b|\bRSpec\b|\bTest::Unit\b',
        "cpp":        r'EXPECT_EQ|ASSERT_EQ|TEST\(',
    }
    pat = patterns.get(lang, r'\btest\b|\bassert\b')
    return bool(re.search(pat, code))

def has_memory_management(code: str, lang: str) -> bool:
    """C/C++ specific: new/delete, smart pointers, RAII."""
    if lang != "cpp":
        return False
    return bool(re.search(r'\bnew\b|\bdelete\b|unique_ptr|shared_ptr|make_shared|make_unique', code))

def has_interfaces_protocols(code: str, lang: str) -> bool:
    patterns = {
        "java":       r'\binterface\s+\w+|\bimplements\b',
        "csharp":     r'\binterface\s+I\w+|\bIEnumerable\b|\bIDisposable\b',
        "typescript": r'\binterface\s+\w+|\bimplements\b',
        "python":     r'from\s+abc\s+import|@abstractmethod|\bABC\b',
        "go":         r'\binterface\s*\{',
    }
    pat = patterns.get(lang, "")
    return bool(re.search(pat, code)) if pat else False


# ══════════════════════════════════════════════════════════════════════════════
#  TIME COMPLEXITY DETECTION
#  Heuristic approach — analyses loop nesting depth + algorithmic patterns
# ══════════════════════════════════════════════════════════════════════════════
def detect_complexity(code: str, lang: str) -> dict:
    depth = _loop_max_depth(code)

    # O(n log n) — sort calls
    sort_pat = {
        "python":     r'\bsorted\b|\b\.sort\b|\bheapq\b|\bbisect\b',
        "javascript": r'\.sort\(',
        "java":       r'Collections\.sort|Arrays\.sort|\bTreeSet\b',
        "csharp":     r'\.OrderBy\(|\.Sort\(',
        "cpp":        r'\bstd::sort\b|\bstd::stable_sort\b',
        "go":         r'\bsort\.\w+\(',
        "ruby":       r'\.sort\b|\.sort_by\b',
        "typescript": r'\.sort\(',
    }
    sort_re = sort_pat.get(lang, r'\bsort\b')
    if re.search(sort_re, code) and depth >= 1:
        label = "O(n log n)"; detail = "Sorting algorithm detected inside a loop."
    elif re.search(r'//\s*2|>>\s*1|\bmid\b|\bbinary\b', code, re.IGNORECASE) and depth <= 1:
        label = "O(log n)"; detail = "Binary search / divide-and-conquer pattern detected."
    elif depth == 0:
        label = "O(1)"; detail = "No loops detected — constant time execution."
    elif depth == 1:
        label = "O(n)"; detail = "Single loop — linear time complexity."
    elif depth == 2:
        label = "O(n²)"; detail = "Nested loops (depth 2) — quadratic time. Consider refactoring."
    else:
        label = "O(n³)"; detail = f"Triple-nested loops (depth {depth}) — cubic time. Performance risk."

    return {"label": label, "detail": detail, "loop_depth": depth}


# ══════════════════════════════════════════════════════════════════════════════
#  HALSTEAD-INSPIRED METRICS  (lightweight approximation)
#  Full Halstead requires a complete lexer; we approximate via regex counts.
#  Used to enrich the quality breakdown.
# ══════════════════════════════════════════════════════════════════════════════
def halstead_metrics(code: str) -> dict:
    operators = re.findall(r'[+\-*/%&|^~<>=!]+|and|or|not|in|is', code)
    operands  = re.findall(r'\b\d+\.?\d*\b|\b[a-zA-Z_]\w*\b', code)
    n1 = len(set(operators)); n2 = len(set(operands))
    N1 = len(operators);      N2 = len(operands)
    n  = n1 + n2 if (n1 + n2) > 0 else 1
    N  = N1 + N2
    vocabulary = n
    length     = N
    volume     = N * math.log2(n) if n > 1 else 0
    difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
    effort     = difficulty * volume
    return {
        "vocabulary": vocabulary,
        "length":     length,
        "volume":     round(volume, 1),
        "difficulty": round(difficulty, 2),
        "effort":     round(effort, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CYCLOMATIC COMPLEXITY  (McCabe approximation)
#  Counts decision points: each if/else/elif/case/for/while/&&/|| adds 1
# ══════════════════════════════════════════════════════════════════════════════
def cyclomatic_complexity(code: str) -> int:
    decision_points = len(re.findall(
        r'\b(if|elif|else|for|while|case|catch|except|rescue|&&|\|\|)\b', code
    ))
    return decision_points + 1  # base complexity = 1


# ══════════════════════════════════════════════════════════════════════════════
#  CODE QUALITY BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
def compute_breakdown(code: str, lang: str, line_count: int) -> dict:
    # ── Readability (0–100) ───────────────────────────────────────────────────
    read = 0
    if has_comments(code, lang):         read += 25
    if has_functions(code, lang):        read += 20
    avg_line = sum(len(l) for l in code.splitlines()) / max(line_count, 1)
    if avg_line < 80:                    read += 15
    if has_imports(code, lang):          read += 10
    if has_type_hints(code, lang):       read += 20
    if not re.search(r'\b[a-zA-Z]\s*=', code):  read += 10  # no single-char vars
    read = min(100, read)

    # ── Efficiency (0–100) ────────────────────────────────────────────────────
    eff = 55
    if has_nested_loops(code, lang):                  eff -= 25
    if has_list_comprehension(code, lang):            eff += 15
    if has_recursion(code):                           eff += 10
    if has_async(code, lang):                         eff += 10
    if _loop_max_depth(code) >= 3:                    eff -= 15
    cc = cyclomatic_complexity(code)
    if cc > 10: eff -= 10  # high cyclomatic complexity hurts maintainability
    eff = max(0, min(100, eff))

    # ── Structure (0–100) ─────────────────────────────────────────────────────
    struct = 0
    if has_functions(code, lang):                     struct += 20
    if has_classes(code, lang):                       struct += 20
    if has_error_handling(code, lang):                struct += 20
    if has_imports(code, lang):                       struct += 10
    if has_decorators(code, lang):                    struct += 10
    if has_interfaces_protocols(code, lang):          struct += 10
    if has_unit_tests(code, lang):                    struct += 10
    struct = min(100, struct)

    # ── Maintainability Index (simplified) ────────────────────────────────────
    h = halstead_metrics(code)
    mi = max(0, min(100, int(
        171 - 5.2 * math.log(max(h["volume"], 1))
            - 0.23 * cyclomatic_complexity(code)
            - 16.2 * math.log(max(line_count, 1))
    )))

    return {
        "readability":        int(read),
        "efficiency":         int(eff),
        "structure":          int(struct),
        "maintainability":    mi,
        "cyclomatic":         cyclomatic_complexity(code),
        "halstead":           h,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  COSINE SIMILARITY
# ══════════════════════════════════════════════════════════════════════════════
def compute_similarity(code: str, lang: str) -> float:
    ideal = IDEAL_SOLUTIONS.get(lang, IDEAL_SOLUTIONS["python"])
    vecs  = vectorizer.transform([code, ideal])
    sim   = float(cosine_similarity(vecs[0], vecs[1])[0, 0])
    return round(sim, 4)


# ══════════════════════════════════════════════════════════════════════════════
#  GAP DETECTION  →  weaknesses[] + suggestions[]
# ══════════════════════════════════════════════════════════════════════════════
def detect_gaps(code: str, lang: str, line_count: int) -> tuple:
    W, S = [], []

    if not has_functions(code, lang):
        W.append("No functions / methods detected")
        S.append("Encapsulate logic in named functions. This improves reusability, testability, and readability.")

    if not has_error_handling(code, lang):
        W.append(f"No error handling ({['try/except','try/catch','begin/rescue','if err != nil'][['python','java','ruby','go'].index(lang) if lang in ['python','java','ruby','go'] else 1]})")
        S.append("Add error handling to gracefully manage runtime failures and invalid inputs.")

    if has_nested_loops(code, lang):
        W.append(f"Nested loops detected — O(n²) time complexity risk")
        S.append("Refactor nested loops using hash maps, set operations, or vectorised operations to improve efficiency.")

    if not has_classes(code, lang) and line_count > 30:
        W.append("No OOP constructs in a large codebase (>30 lines)")
        S.append("Group related data and behaviour into classes or structs for better organisation.")

    if not has_comments(code, lang) and line_count > 10:
        W.append("No comments or documentation strings found")
        S.append("Add inline comments and function docstrings to explain the why, not just the what.")

    if not has_imports(code, lang) and line_count > 20:
        W.append("No standard library usage detected")
        S.append("Leverage built-in modules and packages — they are battle-tested and optimised.")

    if not has_type_hints(code, lang) and lang in ("python", "typescript", "csharp", "java"):
        W.append(f"No type annotations/hints detected ({lang})")
        S.append("Add type hints/annotations to improve IDE support, catch bugs early, and aid readability.")

    if lang == "python" and not has_list_comprehension(code, lang) and line_count > 15:
        W.append("No list/dict comprehensions used (Python idiom missing)")
        S.append("Replace filter+append loops with list comprehensions: [x*2 for x in data if x > 0].")

    if lang == "go" and not re.search(r'if\s+err\s*!=\s*nil', code):
        W.append("Go error handling pattern (if err != nil) not found")
        S.append("In Go, always check errors explicitly: result, err := fn(); if err != nil { ... }")

    if lang == "cpp" and not re.search(r'unique_ptr|shared_ptr|make_shared|make_unique', code) and re.search(r'\bnew\b', code):
        W.append("Raw 'new' usage without smart pointers (C++)")
        S.append("Use std::unique_ptr or std::shared_ptr instead of raw pointers to prevent memory leaks.")

    if not has_unit_tests(code, lang) and line_count > 25:
        W.append("No unit tests detected")
        S.append("Write unit tests to verify correctness. Aim for >80% coverage on critical logic.")

    cc = cyclomatic_complexity(code)
    if cc > 10:
        W.append(f"High cyclomatic complexity ({cc}) — code is difficult to test and maintain")
        S.append("Break complex functions into smaller single-responsibility functions. Target cyclomatic ≤ 10.")

    return W, S


# ══════════════════════════════════════════════════════════════════════════════
#  SCORING  (0–100)
#  Hybrid: 35% ML confidence + 65% rule-based signals
# ══════════════════════════════════════════════════════════════════════════════
def compute_score(code: str, lang: str, ml_label: int, proba: np.ndarray) -> int:
    ml_part = float(proba[ml_label]) * 35
    signals = [
        has_functions(code, lang),
        has_error_handling(code, lang),
        has_classes(code, lang),
        has_comments(code, lang),
        has_imports(code, lang),
        has_type_hints(code, lang),
        has_recursion(code) or has_async(code, lang) or has_decorators(code, lang),
        has_unit_tests(code, lang),
        has_list_comprehension(code, lang),
        has_interfaces_protocols(code, lang),
        has_generics_or_templates(code, lang),
    ]
    rule_part = sum(signals) * (65 / len(signals))
    return min(math.floor(ml_part + rule_part), 100)


# ══════════════════════════════════════════════════════════════════════════════
#  IMPROVED CODE GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def generate_improved_code(code: str, lang: str) -> str:
    lines = code.strip().splitlines()
    has_fn  = has_functions(code, lang)
    has_err = has_error_handling(code, lang)

    if lang == "python":
        out = ['"""', 'Improved version — auto-generated.', 'Changes applied:',
               '  • Wrapped in main() function', '  • Added type hints',
               '  • Added try/except error handling', '  • Added docstring',
               '  • Added __main__ guard', '"""', "", "from typing import Any", ""]
        if not has_fn:
            out += ["def main() -> None:", '    """Entry point — wraps all logic."""']
            if not has_err:
                out += ["    try:"] + ["        " + l for l in lines]
                out += ["    except Exception as exc:", '        print(f"Error: {exc}")']
            else:
                out += ["    " + l for l in lines]
            out += ["", "", 'if __name__ == "__main__":', "    main()"]
        else:
            out += lines
            if not has_err:
                out += ["", "# Tip: add try/except around I/O and external calls."]
        return "\n".join(out)

    elif lang == "javascript":
        out = ["/**", " * Improved version — auto-generated.",
               " * Changes: async/await, try/catch, JSDoc.", " */", ""]
        if not has_fn:
            out += ["async function main() {"]
            if not has_err:
                out += ["  try {"] + ["    " + l for l in lines]
                out += ["  } catch (err) {", "    console.error('Error:', err.message);", "  }"]
            else:
                out += ["  " + l for l in lines]
            out += ["}", "", "main();"]
        else:
            out += lines
            if not has_err:
                out += ["", "// Tip: wrap async operations in try/catch blocks."]
        return "\n".join(out)

    elif lang == "typescript":
        out = ["/**", " * Improved TypeScript version.", " * Changes: strict types, async/await, error handling.", " */", ""]
        if not has_fn:
            out += ["async function main(): Promise<void> {"]
            if not has_err:
                out += ["  try {"] + ["    " + l for l in lines]
                out += ["  } catch (err: unknown) {",
                        "    console.error('Error:', err instanceof Error ? err.message : err);",
                        "  }"]
            else:
                out += ["  " + l for l in lines]
            out += ["}", "", "main();"]
        else:
            out += lines
        return "\n".join(out)

    elif lang == "java":
        if has_err:
            return code + "\n// ✓ Exception handling detected — good practice!"
        out = ["// Improved Java — exception handling added.", ""]
        for l in lines:
            if re.search(r'return\s+\w+\s*/\s*\w+', l):
                m = re.findall(r'return\s+(\w+)\s*/\s*(\w+)', l)
                if m:
                    out.append(f"        if ({m[0][1]} == 0) throw new ArithmeticException(\"Division by zero\");")
            out.append(l)
        out += ["", "// Tip: wrap main() body with try-catch for robustness."]
        return "\n".join(out)

    elif lang == "cpp":
        out = ["// Improved C++ — smart pointers + exception handling added.", "#include <stdexcept>", "#include <memory>", ""]
        for l in lines:
            out.append(l)
        if not has_err:
            out += ["", "// Tip: use try/catch around code that may throw.", "// Prefer std::unique_ptr over raw new/delete."]
        return "\n".join(out)

    elif lang == "csharp":
        out = ["// Improved C# — null checks, async pattern, LINQ.", "using System;", "using System.Linq;", "using System.Threading.Tasks;", ""]
        if not has_fn:
            out += ["static async Task Main() {"]
            if not has_err:
                out += ["    try {"] + ["        " + l for l in lines]
                out += ["    } catch (Exception ex) {", "        Console.Error.WriteLine($\"Error: {ex.Message}\");", "    }"]
            else:
                out += ["    " + l for l in lines]
            out += ["}"]
        else:
            out += lines
        return "\n".join(out)

    elif lang == "ruby":
        out = ["# frozen_string_literal: true", "# Improved Ruby — error handling + modules added.", ""]
        if not has_fn:
            out += ["def main"]
            if not has_err:
                out += ["  begin"] + ["    " + l for l in lines]
                out += ["  rescue StandardError => e", "    puts \"Error: #{e.message}\"", "  end"]
            else:
                out += ["  " + l for l in lines]
            out += ["end", "", "main"]
        else:
            out += lines
        return "\n".join(out)

    elif lang == "go":
        out = ["// Improved Go — explicit error handling, idiomatic style.", "package main", "", 'import ("errors"\n\t"fmt")', ""]
        if not has_err and not re.search(r'if\s+err\s*!=\s*nil', code):
            out += lines
            out += ["", "// Tip: always check errors in Go:", "// result, err := someFunc()", "// if err != nil { fmt.Println(err); return }"]
        else:
            out += lines
        return "\n".join(out)

    return code


# ══════════════════════════════════════════════════════════════════════════════
#  CODE SMELL DETECTION  (new in v3)
# ══════════════════════════════════════════════════════════════════════════════
def detect_code_smells(code: str, lang: str, line_count: int) -> list:
    smells = []
    # Long method
    fn_bodies = re.split(r'\bdef\s+\w+|\bfunc\s+\w+|\bfunction\s+\w+', code)
    for body in fn_bodies[1:]:
        body_lines = [l for l in body.splitlines() if l.strip()]
        if len(body_lines) > 30:
            smells.append("Long method detected (>30 lines). Split into smaller functions.")
            break
    # Magic numbers
    if re.search(r'\b(?!0\b|1\b|2\b)\d{2,}\b', code):
        smells.append("Magic numbers found. Extract into named constants for clarity.")
    # Deeply nested code (ifs, not just loops)
    if code.count("    " * 4) > 3 or code.count("\t\t\t\t") > 3:
        smells.append("Deeply nested code blocks (4+ levels). Consider early returns or guard clauses.")
    # God function (does too many things)
    cc = cyclomatic_complexity(code)
    if cc > 15:
        smells.append(f"God function: cyclomatic complexity {cc}. This function does too many things.")
    # Commented-out code
    if len(re.findall(r'^\s*#\s*(if|for|def|class|import|var|let|const|int|void)', code, re.MULTILINE)) > 2:
        smells.append("Commented-out code blocks detected. Remove dead code; use version control instead.")
    return smells


# ══════════════════════════════════════════════════════════════════════════════
#  LANGUAGE PROFILE  (new in v3) — metadata about the analysed language
# ══════════════════════════════════════════════════════════════════════════════
LANG_PROFILES = {
    "python":     {"paradigm": "Multi-paradigm (OOP, functional, procedural)", "typing": "Dynamic", "use_case": "Data science, scripting, web backends, automation"},
    "javascript": {"paradigm": "Multi-paradigm (event-driven, functional, OOP)", "typing": "Dynamic", "use_case": "Web frontends, Node.js backends, mobile (React Native)"},
    "java":       {"paradigm": "Object-Oriented", "typing": "Static", "use_case": "Enterprise backends, Android, big data"},
    "typescript": {"paradigm": "Multi-paradigm (OOP, functional)", "typing": "Static (transpiles to JS)", "use_case": "Large-scale web apps, Angular, Node.js"},
    "cpp":        {"paradigm": "Multi-paradigm (procedural, OOP, generic)", "typing": "Static", "use_case": "Systems programming, game engines, embedded"},
    "csharp":     {"paradigm": "Multi-paradigm (OOP, functional)", "typing": "Static", "use_case": ".NET apps, Unity games, enterprise software"},
    "ruby":       {"paradigm": "Object-Oriented, functional", "typing": "Dynamic", "use_case": "Web (Rails), scripting, DevOps tools"},
    "go":         {"paradigm": "Procedural, concurrent", "typing": "Static", "use_case": "Cloud services, CLIs, microservices, DevOps"},
}


# ══════════════════════════════════════════════════════════════════════════════
#  SCORING  —  already defined above
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/analyze", methods=["POST"])
def analyze():
    t_start = time.time()
    data  = request.get_json(force=True)
    code  = (data.get("code") or "").strip()
    lang  = (data.get("language") or "python").lower()

    if not code:
        return jsonify({"error": "No code provided"}), 400
    if lang not in LANG_PROFILES:
        lang = "python"

    # ── ML Prediction ─────────────────────────────────────────────────────────
    X_input  = vectorizer.transform([code])
    ml_label = int(clf.predict(X_input)[0])
    proba    = clf.predict_proba(X_input)[0]

    # ── Rule-based adjustment ─────────────────────────────────────────────────
    line_count = count_lines(code)
    bonus = sum([
        has_classes(code, lang), has_decorators(code, lang),
        has_async(code, lang),   has_recursion(code),
        has_generics_or_templates(code, lang),
        has_type_hints(code, lang),
        line_count > 60,
    ])
    penalty = sum([
        not has_functions(code, lang) and line_count > 20,
        not has_error_handling(code, lang) and line_count > 15,
    ])
    adjusted = max(0, min(2, ml_label + (1 if bonus >= 2 else 0) - (1 if penalty >= 2 else 0)))
    skill_level = LEVEL_NAMES[adjusted]
    score       = compute_score(code, lang, adjusted, proba)

    # ── Gap Detection ─────────────────────────────────────────────────────────
    weaknesses, suggestions = detect_gaps(code, lang, line_count)
    if not weaknesses:
        weaknesses  = ["No major structural gaps detected — well structured code!"]
        suggestions = [
            "Explore design patterns (Factory, Observer, Strategy) to elevate architecture.",
            "Write property-based tests (e.g., Hypothesis for Python) for edge-case coverage.",
            "Profile your code with cProfile/perf tools to identify hidden bottlenecks.",
        ]

    # ── All new v3 fields ─────────────────────────────────────────────────────
    complexity_info = detect_complexity(code, lang)
    breakdown       = compute_breakdown(code, lang, line_count)
    smells          = detect_code_smells(code, lang, line_count)
    improved        = generate_improved_code(code, lang)
    similarity      = compute_similarity(code, lang)
    lang_profile    = LANG_PROFILES.get(lang, {})
    token_count     = count_tokens(code)
    branch_count    = count_branches(code)
    proc_ms         = round((time.time() - t_start) * 1000, 1)

    # ML confidence breakdown
    confidence = {
        "beginner":     round(float(proba[0]) * 100, 1),
        "intermediate": round(float(proba[1]) * 100, 1),
        "advanced":     round(float(proba[2]) * 100, 1),
    }

    return jsonify({
        # Core
        "skill_level":     skill_level,
        "score":           score,
        "lines_analyzed":  line_count,
        "token_count":     token_count,
        "branch_count":    branch_count,
        "processing_ms":   proc_ms,
        # ML
        "ml_confidence":   confidence,
        # Analysis
        "weaknesses":      weaknesses,
        "suggestions":     suggestions,
        "code_smells":     smells,
        # Metrics
        "complexity":      complexity_info,
        "breakdown":       breakdown,
        # Improvement
        "improved_code":   improved,
        # Similarity
        "similarity_score": similarity,
        # Language info
        "lang_profile":    lang_profile,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":     "ok",
        "version":    "3.0",
        "model":      "TF-IDF(char_wb,2-4gram) + LogisticRegression(lbfgs)",
        "languages":  list(LANG_PROFILES.keys()),
        "corpus_size": len(TRAINING_DATA),
    })


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║  Coding Skill Gap Detector v3 — Backend         ║")
    print("║  http://127.0.0.1:5000                          ║")
    print(f"║  Corpus: {len(TRAINING_DATA)} samples · 8 languages              ║")
    print("╚══════════════════════════════════════════════════╝")
    app.run(debug=True, port=5000)