from bfcl_eval.model_handler.utils import ast_parse

sample = """[
   {
      "name":"solve_quadratic",
      "arguments":{
         "type":"all",
         "a":3,
         "b":-11,
         "c":-4
      }
   }
]
"""

sample = """[
   {
      "solve_quadratic_equation":{
         "a":[
            2
         ],
         "b":[
            6
         ],
         "c":[
            5
         ]
      }
   }
]
"""

result = ast_parse(sample)
print(result)
