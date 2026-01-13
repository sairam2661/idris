import re
from pathlib import Path 
import uuid 

def extract_functions(test_dir, max_count=8000):
    functions = []
    name_regex = r"(@[\w\d._-]+)"
    for ll_file in Path(test_dir).rglob("*.ll"):
        try:
            content = ll_file.read_text()
        except OSError:
            continue
        
        lines = content.split('\n')
        func_lines = []
        in_func = False
        brace_count = 0
        
        for line in lines:
            if line.strip().startswith('define '):
                in_func = True
                func_lines = [line]
                brace_count = line.count('{') - line.count('}')
            elif in_func:
                func_lines.append(line)
                brace_count += line.count('{') - line.count('}')
                if brace_count <= 0:
                    func = '\n'.join(func_lines)
                    if 100 < len(func) < 2000:
                        unique_id = uuid.uuid4().hex[:6]
                        new_name = f"@fuzz_target_{unique_id}"
                        func = re.sub(name_regex, new_name, func, count=1)
                        functions.append(func)
                    in_func = False
                    if len(functions) >= max_count:
                        return functions
    return functions