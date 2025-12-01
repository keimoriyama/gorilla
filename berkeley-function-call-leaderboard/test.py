from bfcl_eval.model_handler.utils import ast_parse

sample = """
solve(a=1, v=2)
""".strip()

result = ast_parse(sample)
